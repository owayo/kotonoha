"""JSUTラベルノイズ診断スクリプト.

辞書アクセント型と学習データのラベルを比較し、
不一致のケースを検出・レポートする。

アクセント句の先頭内容語のみを比較対象とする。
（句内の機能語や後続内容語はアクセント句全体の型を共有するだけなので除外）

Usage:
  uv run python diagnose_label_noise.py
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

# 機能語POS（アクセント句の頭にならない品詞）
_FUNCTION_POS = {"助詞", "助動詞"}


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


def _parse_dict_accent(val: str) -> int | None:
    """dict_accent_typeをintに変換。無効なら None.

    Returns:
        アクセント型 or None.

    """
    if val in ("*", "", "None"):
        return None
    m = re.match(r'^"?(\d+)"?$', val)
    if m:
        return int(m.group(1))
    return None


def _count_morae(reading: str) -> int:
    """カタカナ読みからモーラ数を数える.

    Returns:
        モーラ数.

    """
    count = 0
    for ch in reading:
        if ch in "ァィゥェォャュョヮ":
            continue  # 拗音の小文字はモーラに数えない
        if ch == "ー":
            count += 1
        elif "\u30a0" <= ch <= "\u30ff":
            count += 1
    return count


def _segment_accent_phrases(
    morphemes: list[dict], accent_phrases: list[dict]
) -> list[list[dict]]:
    """形態素列をアクセント句ごとにグループ化する.

    accent_phrasesのmora_countを使って境界を決定。

    Returns:
        アクセント句ごとの形態素リスト.

    """
    result: list[list[dict]] = []
    mi = 0  # morpheme index
    for ap in accent_phrases:
        target_morae = ap["mora_count"]
        group: list[dict] = []
        current_morae = 0
        while mi < len(morphemes) and current_morae < target_morae:
            morph = morphemes[mi]
            pron = morph.get("pronunciation", morph.get("reading", ""))
            current_morae += _count_morae(pron)
            group.append(morph)
            mi += 1
        result.append(group)
    # 残りがあれば最後のグループに追加
    while mi < len(morphemes):
        if result:
            result[-1].append(morphemes[mi])
        mi += 1
    return result


def main() -> None:
    """診断を実行する."""
    _load_dotenv()

    jsut_path = Path(os.environ.get("FINETUNE_DATA", ""))
    print(f"Loading: {jsut_path}")
    with open(jsut_path, encoding="utf-8") as f:
        data = json.load(f)

    utterances = data["utterances"]
    print(f"Utterances: {len(utterances)}")

    # 統計
    total_phrases = 0
    head_with_dict = 0
    match = 0
    mismatch = 0

    mismatch_examples: list[dict] = []
    mismatch_by_pos: Counter[str] = Counter()
    mismatch_pattern: Counter[str] = Counter()

    # モーラ数一致による信頼度チェック
    mono_morph_match = 0  # 1形態素のみのアクセント句で一致
    mono_morph_mismatch = 0

    for utt in utterances:
        morphemes = utt.get("morphemes", [])
        accent_phrases = utt.get("accent_phrases", [])

        if not accent_phrases:
            continue

        groups = _segment_accent_phrases(morphemes, accent_phrases)

        for group, ap in zip(groups, accent_phrases, strict=True):
            total_phrases += 1
            phrase_accent = ap["accent_type"]

            # 先頭内容語を探す
            head = None
            for morph in group:
                if morph.get("pos", "") not in _FUNCTION_POS:
                    head = morph
                    break

            if head is None:
                continue

            dict_val = head.get("dict_accent_type", "*")
            dict_at = _parse_dict_accent(dict_val)

            if dict_at is None:
                continue

            head_with_dict += 1
            is_single_content = (
                sum(1 for m in group if m.get("pos", "") not in _FUNCTION_POS) == 1
            )

            if dict_at == phrase_accent:
                match += 1
                if is_single_content:
                    mono_morph_match += 1
            else:
                mismatch += 1
                if is_single_content:
                    mono_morph_mismatch += 1

                pos = head.get("pos", "")
                mismatch_by_pos[pos] += 1
                pattern = f"dict={dict_at}->label={phrase_accent}"
                mismatch_pattern[pattern] += 1

                if len(mismatch_examples) < 40:
                    phrase_text = "".join(m.get("surface", "") for m in group)
                    mismatch_examples.append(
                        {
                            "utt_id": utt.get("utterance_id", "?"),
                            "phrase": phrase_text,
                            "head_surface": head.get("surface", ""),
                            "head_reading": head.get("reading", ""),
                            "pos": pos,
                            "dict_accent": dict_at,
                            "phrase_accent": phrase_accent,
                            "is_single": is_single_content,
                            "morph_count": len(group),
                        }
                    )

    # レポート
    print(f"\n{'=' * 60}")
    print("Label Noise Diagnosis Report (Accent Phrase Level)")
    print(f"{'=' * 60}")
    print(f"\n  Accent phrases total:    {total_phrases:,}")
    print(f"  Head word with dict:     {head_with_dict:,}")
    print(f"  Match:                   {match:,}")
    print(f"  Mismatch:                {mismatch:,}")
    if head_with_dict > 0:
        match_pct = match / head_with_dict * 100
        mismatch_pct = mismatch / head_with_dict * 100
        print(f"  Match rate:              {match_pct:.1f}%")
        print(f"  Mismatch rate:           {mismatch_pct:.1f}%")

    print("\n  Single-content-word phrases:")
    print(f"    Match:    {mono_morph_match:,}")
    print(f"    Mismatch: {mono_morph_mismatch:,}")
    if mono_morph_match + mono_morph_mismatch > 0:
        mono_total = mono_morph_match + mono_morph_mismatch
        print(f"    Match rate: {mono_morph_match / mono_total * 100:.1f}%")
        print("    (These are the most reliable noise indicators)")

    print("\n  Mismatch by POS:")
    for pos, cnt in mismatch_by_pos.most_common():
        print(f"    {pos}: {cnt}")

    print("\n  Top mismatch patterns (dict->phrase_label):")
    for pattern, cnt in mismatch_pattern.most_common(15):
        print(f"    {pattern}: {cnt}")

    print("\n  Example mismatches:")
    for ex in mismatch_examples[:25]:
        single_mark = " [SINGLE]" if ex["is_single"] else ""
        print(
            f"    [{ex['utt_id']}] "
            f"「{ex['phrase']}」 "
            f"head={ex['head_surface']}({ex['head_reading']}) "
            f"{ex['pos']} "
            f"dict={ex['dict_accent']} "
            f"label={ex['phrase_accent']}"
            f"{single_mark}"
        )

    # 分析
    print(f"\n{'=' * 60}")
    print("Analysis")
    print(f"{'=' * 60}")
    flat_to_nonflat = sum(
        v for k, v in mismatch_pattern.items() if k.startswith("dict=0->")
    )
    nonflat_mismatch = mismatch - flat_to_nonflat
    print(
        f"\n  dict=0 -> label!=0 (アクセント結合による変化の可能性): {flat_to_nonflat}"
    )
    print(f"  dict!=0 -> label!=dict (潜在的ラベルエラー): {nonflat_mismatch}")

    if mono_morph_mismatch > 0:
        print(f"\n  高信頼ノイズ候補 (1内容語のみの句で不一致): {mono_morph_mismatch}")
        print("  → これらは句結合の影響が少なく、ラベルエラーの可能性が高い")


if __name__ == "__main__":
    main()
