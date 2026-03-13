"""LLM生成コーパスを kotonoha 学習データ形式に変換する.

chain_flag ベースのデータを accent_phrases 付きの形式に変換し、
アクセント辞書で dict_accent_type を補完し、品質検証を行う。

Usage:
  uv run python convert_corpus.py
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

_SMALL_KANA = set("ァィゥェォャュョヮ")


def _load_dotenv() -> None:
    """Load .env file."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _count_morae(reading: str) -> int:
    """カタカナ読みからモーラ数を算出する.

    Returns:
        モーラ数.

    """
    count = 0
    for ch in reading:
        if ch in _SMALL_KANA:
            continue
        if ch == "ー":
            count += 1
        elif "\u30a0" <= ch <= "\u30ff":
            count += 1
    return max(count, 1)


def _load_accent_dicts(
    paths: list[Path],
) -> dict[tuple[str, str], str]:
    """複数のアクセント辞書を読み込みマージする.

    Returns:
        (lemma, reading) → accent_type の辞書.

    """
    merged: dict[tuple[str, str], str] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                lemma, reading, accent = row[0], row[1], row[2]
                if lemma in ("lemma", "#") or lemma.startswith("#"):
                    continue
                merged[(lemma, reading)] = accent
        print(f"  {path.name}: {len(merged)} entries (cumulative)")
    return merged


def _build_accent_phrases(morphemes: list[dict]) -> list[dict]:
    """chain_flag からアクセント句リストを構築する.

    Returns:
        accent_phrases のリスト.

    """
    phrases: list[dict] = []
    current_morae = 0
    current_accent = 0

    for m in morphemes:
        pron = m.get("pronunciation", m.get("reading", ""))
        morae = _count_morae(pron)
        chain = m.get("chain_flag", 0)

        if chain == 0 and (phrases or current_morae > 0):
            phrases.append({"accent_type": current_accent, "mora_count": current_morae})
            current_morae = 0

        current_accent = m.get("accent_type", 0)
        current_morae += morae

    if current_morae > 0:
        phrases.append({"accent_type": current_accent, "mora_count": current_morae})

    return phrases


def _validate_utterance(utt: dict) -> list[str]:
    """発話データの品質検証を行う.

    Returns:
        エラーメッセージのリスト（空なら合格）.

    """
    errors: list[str] = []
    morphemes = utt.get("morphemes", [])
    accent_phrases = utt.get("accent_phrases", [])
    text = utt.get("text", "")

    if not morphemes:
        errors.append("no morphemes")
        return errors

    # テキスト復元チェック
    reconstructed = "".join(m["surface"] for m in morphemes)
    # 句読点や記号の差異は許容
    clean_text = re.sub(r"[。、！？!?,.\s]", "", text)
    clean_recon = re.sub(r"[。、！？!?,.\s]", "", reconstructed)
    if clean_text != clean_recon:
        errors.append(f"text mismatch: '{text[:30]}' vs '{reconstructed[:30]}'")

    # accent_type 範囲チェック
    for ap in accent_phrases:
        if ap["accent_type"] > ap["mora_count"]:
            errors.append(
                f"accent_type {ap['accent_type']} > mora_count {ap['mora_count']}"
            )

    # accent_type 一貫性チェック（同一句内で同じ値か）
    phrase_idx = 0
    mora_remaining = accent_phrases[0]["mora_count"] if accent_phrases else 0
    expected_accent = accent_phrases[0]["accent_type"] if accent_phrases else -1

    for m in morphemes:
        at = m.get("accent_type", -1)
        if at != expected_accent and mora_remaining > 0:
            errors.append(
                f"inconsistent accent_type in phrase: "
                f"'{m['surface']}' has {at}, expected {expected_accent}"
            )
            break
        pron = m.get("pronunciation", m.get("reading", ""))
        mora_remaining -= _count_morae(pron)
        if mora_remaining <= 0 and phrase_idx + 1 < len(accent_phrases):
            phrase_idx += 1
            mora_remaining = accent_phrases[phrase_idx]["mora_count"]
            expected_accent = accent_phrases[phrase_idx]["accent_type"]

    # 品詞チェック
    valid_pos = {
        "名詞",
        "動詞",
        "形容詞",
        "副詞",
        "連体詞",
        "接続詞",
        "感動詞",
        "助詞",
        "助動詞",
        "接頭辞",
        "接尾辞",
        "代名詞",
        "形状詞",
        "記号",
        "補助記号",
    }
    for m in morphemes:
        if m["pos"] not in valid_pos:
            errors.append(f"unknown pos: '{m['pos']}'")
            break

    # reading 存在チェック
    for m in morphemes:
        if not m.get("reading"):
            errors.append(f"missing reading: '{m['surface']}'")
            break

    return errors


def main() -> None:
    """コーパス変換を実行する."""
    _load_dotenv()

    # 入力
    corpus_path = Path(
        os.environ.get(
            "CORPUS_DATA",
            "/mnt/c/GitHub/kotonoha-training-data/train/corpus.json",
        )
    )
    print(f"Loading: {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        raw_utts = raw_data
    else:
        raw_utts = raw_data.get("utterances", [])
    print(f"  Raw utterances: {len(raw_utts)}")

    # アクセント辞書
    dict_str = os.environ.get("ACCENT_DICT", "")
    dict_paths = [Path(p) for p in dict_str.split(":") if p]
    accent_dict: dict[tuple[str, str], str] = {}
    if dict_paths:
        print("\nLoading accent dicts:")
        accent_dict = _load_accent_dicts(dict_paths)

    # 変換
    print("\nConverting...")
    converted = []
    errors_by_type: Counter[str] = Counter()
    rejected = 0
    enriched_count = 0

    for i, raw in enumerate(raw_utts):
        utt_id = f"CORPUS_{i + 1:04d}"
        morphemes = raw.get("morphemes", [])

        # 不足フィールドの補完
        for m in morphemes:
            if "pos_detail3" not in m:
                m["pos_detail3"] = "*"
            if "pronunciation" not in m:
                m["pronunciation"] = m.get("reading", "")
            if "dict_accent_type" not in m:
                lemma = m.get("lemma", "")
                reading = m.get("reading", "")
                key = (lemma, reading)
                if key in accent_dict:
                    m["dict_accent_type"] = accent_dict[key]
                    enriched_count += 1
                else:
                    base = lemma.split("-")[0] if "-" in lemma else lemma
                    key2 = (base, reading)
                    if key2 in accent_dict:
                        m["dict_accent_type"] = accent_dict[key2]
                        enriched_count += 1
                    else:
                        m["dict_accent_type"] = "*"

        # accent_phrases を chain_flag から構築
        accent_phrases = _build_accent_phrases(morphemes)

        # chain_flag を除去（kotonoha では不使用）
        for m in morphemes:
            m.pop("chain_flag", None)

        utt = {
            "utterance_id": utt_id,
            "text": raw.get("text", ""),
            "morphemes": morphemes,
            "accent_phrases": accent_phrases,
        }

        # 検証
        errs = _validate_utterance(utt)
        if errs:
            rejected += 1
            for e in errs:
                errors_by_type[e.split(":")[0]] += 1
            if rejected <= 10:
                print(f"  REJECT [{utt_id}] {raw.get('text', '')[:40]}")
                for e in errs:
                    print(f"    - {e}")
        else:
            converted.append(utt)

    # 出力
    output = {"utterances": converted}
    out_path = corpus_path.parent / "corpus_converted.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # レポート
    total_morphs = sum(len(u["morphemes"]) for u in converted)
    print(f"\n{'=' * 60}")
    print("Conversion Report")
    print(f"{'=' * 60}")
    print(f"  Input:     {len(raw_utts)} utterances")
    print(f"  Accepted:  {len(converted)} utterances")
    print(f"  Rejected:  {rejected} utterances")
    print(f"  Morphemes: {total_morphs}")
    print(f"  Enriched dict_accent_type: {enriched_count}")
    print(f"  Output:    {out_path}")

    if errors_by_type:
        print("\n  Rejection reasons:")
        for reason, cnt in errors_by_type.most_common():
            print(f"    {reason}: {cnt}")

    # accent_type 分布
    at_dist: Counter[int] = Counter()
    for u in converted:
        for ap in u["accent_phrases"]:
            at_dist[ap["accent_type"]] += 1
    print("\n  Accent type distribution (phrases):")
    for k in sorted(at_dist):
        print(f"    {k}: {at_dist[k]}")


if __name__ == "__main__":
    main()
