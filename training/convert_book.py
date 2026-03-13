"""book_leader_raw.json を kotonoha 学習データ形式に変換する.

accent_type=0 の句を除外し、違反チェックを行い、
学習に使える形式に変換する。

Usage:
  uv run python convert_book.py
"""

from __future__ import annotations

import csv
import json
import os
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
    return max(count, 1) if reading else 0


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


def _split_by_accent_phrases(utt: dict) -> list[dict]:
    """発話をアクセント句境界で分割し、accent_type=0 の句を除外する.

    accent_phrases の mora_count を使って形態素をグループ化し、
    accent_type > 0 の連続した句のみを保持する。

    Returns:
        分割・フィルタ後の発話リスト（複数になることがある）.

    """
    morphemes = utt.get("morphemes", [])
    accent_phrases = utt.get("accent_phrases", [])

    if not accent_phrases or not morphemes:
        return []

    # 形態素をアクセント句ごとにグループ化
    groups: list[tuple[dict, list[dict]]] = []
    mi = 0
    for ap in accent_phrases:
        target_morae = ap["mora_count"]
        group: list[dict] = []
        current_morae = 0
        while mi < len(morphemes) and current_morae < target_morae:
            m = morphemes[mi]
            pron = m.get("pronunciation", m.get("reading", ""))
            current_morae += _count_morae(pron)
            group.append(m)
            mi += 1
        if ap["mora_count"] > 0:
            groups.append((ap, group))

    # accent_type > 0 の連続区間を抽出
    segments: list[list[tuple[dict, list[dict]]]] = []
    current_segment: list[tuple[dict, list[dict]]] = []

    for ap, group in groups:
        if ap["accent_type"] > 0:
            current_segment.append((ap, group))
        else:
            if len(current_segment) >= 2:
                segments.append(current_segment)
            current_segment = []

    if len(current_segment) >= 2:
        segments.append(current_segment)

    # 各セグメントを発話に変換
    results: list[dict] = []
    for segment in segments:
        seg_morphemes: list[dict] = []
        seg_accent_phrases: list[dict] = []
        for ap, group in segment:
            seg_accent_phrases.append(ap)
            seg_morphemes.extend(group)

        if len(seg_morphemes) >= 3:
            text = "".join(m["surface"] for m in seg_morphemes)
            results.append(
                {
                    "text": text,
                    "morphemes": seg_morphemes,
                    "accent_phrases": seg_accent_phrases,
                }
            )

    return results


def main() -> None:
    """変換を実行する."""
    _load_dotenv()

    input_path = Path("/mnt/c/GitHub/kotonoha-training-data/raw/book_leader_raw.json")
    print(f"Loading: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    utts = data.get("utterances", data if isinstance(data, list) else [])
    print(f"  Raw utterances: {len(utts)}")

    # アクセント辞書
    dict_str = os.environ.get("ACCENT_DICT", "")
    dict_paths = [Path(p) for p in dict_str.split(":") if p]
    accent_dict: dict[tuple[str, str], str] = {}
    if dict_paths:
        print("\nLoading accent dicts:")
        accent_dict = _load_accent_dicts(dict_paths)

    # 変換
    print("\nConverting...")
    converted: list[dict] = []
    total_removed_phrases = 0
    total_violations_fixed = 0
    idx = 0

    for utt in utts:
        # accent_type=0 の句を数える
        zero_count = sum(
            1 for ap in utt.get("accent_phrases", []) if ap["accent_type"] == 0
        )
        total_removed_phrases += zero_count

        # 分割・フィルタ
        segments = _split_by_accent_phrases(utt)

        for seg in segments:
            # accent_type > mora_count の修正
            for ap in seg["accent_phrases"]:
                if ap["accent_type"] > ap["mora_count"]:
                    ap["accent_type"] = ap["mora_count"]
                    total_violations_fixed += 1
                    # 対応する形態素も修正
                    for m in seg["morphemes"]:
                        if m["accent_type"] > ap["mora_count"]:
                            m["accent_type"] = ap["mora_count"]

            # 形態素の accent_type をアクセント句と整合させる
            mi = 0
            for ap in seg["accent_phrases"]:
                target_morae = ap["mora_count"]
                current_morae = 0
                while mi < len(seg["morphemes"]) and current_morae < target_morae:
                    seg["morphemes"][mi]["accent_type"] = ap["accent_type"]
                    pron = seg["morphemes"][mi].get(
                        "pronunciation",
                        seg["morphemes"][mi].get("reading", ""),
                    )
                    current_morae += _count_morae(pron)
                    mi += 1

            # dict_accent_type 補完
            for m in seg["morphemes"]:
                if m.get("dict_accent_type") in ("*", "", None):
                    lemma = m.get("lemma", "")
                    reading = m.get("reading", "")
                    key = (lemma, reading)
                    if key in accent_dict:
                        m["dict_accent_type"] = accent_dict[key]
                    else:
                        base = lemma.split("-")[0] if "-" in lemma else lemma
                        key2 = (base, reading)
                        if key2 in accent_dict:
                            m["dict_accent_type"] = accent_dict[key2]

            idx += 1
            seg["utterance_id"] = f"BOOK_LEADER_{idx:04d}"
            converted.append(seg)

    # 出力
    output = {"utterances": converted}
    out_path = Path("/mnt/c/GitHub/kotonoha-training-data/train/book_leader.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # レポート
    total_morphs = sum(len(u["morphemes"]) for u in converted)
    ap_dist: Counter[int] = Counter()
    for u in converted:
        for ap in u["accent_phrases"]:
            ap_dist[ap["accent_type"]] += 1

    total_ap = sum(ap_dist.values())

    print(f"\n{'=' * 60}")
    print("Conversion Report")
    print(f"{'=' * 60}")
    print(f"  Input:            {len(utts)} utterances")
    print(f"  Output:           {len(converted)} segments")
    print(f"  Morphemes:        {total_morphs}")
    print(f"  Accent phrases:   {total_ap}")
    print(f"  Removed (type=0): {total_removed_phrases} phrases")
    print(f"  Fixed violations: {total_violations_fixed}")
    print(f"  Output:           {out_path}")

    print("\n  Accent type distribution:")
    for k in sorted(ap_dist):
        pct = ap_dist[k] / total_ap * 100
        print(f"    {k:3d}: {ap_dist[k]:5d} ({pct:5.1f}%)")

    type3plus = sum(ap_dist.get(k, 0) for k in range(3, 21))
    print(f"\n  accent_type >= 3: {type3plus / total_ap * 100:.1f}%")

    avg_m = total_morphs / max(len(converted), 1)
    print(f"  Avg morphemes/segment: {avg_m:.1f}")


if __name__ == "__main__":
    main()
