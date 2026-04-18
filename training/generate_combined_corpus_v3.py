"""v23用 複合コーパス生成: 長文化 + 多様化.

v20のcombined_corpus.jsonからの改善点:
- recombined を 3-6 phrases に長文化 (avg 11.4→~14 morphemes、JSUTの16.4に近づける)
- 2つの乱数seedで生成し多様化 (同一seed由来の偏り軽減)
- LLM corpus + phrase-replaced は維持

Usage:
    cd training
    uv run python generate_combined_corpus_v3.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


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


def _extract_accent_phrases(utterances: list[dict]) -> list[list[dict]]:
    """発話リストからアクセント句を抽出する.

    Returns:
        アクセント句のリスト.

    """
    phrases = []
    for utt in utterances:
        morphemes = utt.get("morphemes", [])
        if not morphemes:
            continue
        current: list[dict] = [morphemes[0]]
        for m in morphemes[1:]:
            if m["accent_type"] == current[0]["accent_type"]:
                current.append(m)
            else:
                if current:
                    phrases.append(current)
                current = [m]
        if current:
            phrases.append(current)
    return phrases


def _extract_phrase_groups(utterances: list[dict]) -> list[list[list[dict]]]:
    """各発話をアクセント句のリストに分解する.

    Returns:
        発話ごとのアクセント句リストのリスト.

    """
    result = []
    for utt in utterances:
        morphemes = utt.get("morphemes", [])
        if not morphemes:
            continue
        phrases_list: list[list[dict]] = []
        current: list[dict] = [morphemes[0]]
        for m in morphemes[1:]:
            if m["accent_type"] == current[0]["accent_type"]:
                current.append(m)
            else:
                phrases_list.append(current)
                current = [m]
        if current:
            phrases_list.append(current)
        if len(phrases_list) >= 2:
            result.append(phrases_list)
    return result


def _generate_recombined(
    phrases: list[list[dict]],
    num: int,
    min_phrases: int,
    max_phrases: int,
    rng: random.Random,
    prefix: str,
) -> list[dict]:
    """アクセント句リコンビネーション.

    Returns:
        生成された発話リスト.

    """
    generated = []
    for i in range(num):
        n = rng.randint(min_phrases, max_phrases)
        selected = rng.sample(phrases, min(n, len(phrases)))
        morphemes = []
        for phrase in selected:
            morphemes.extend(phrase)
        if morphemes:
            generated.append(
                {
                    "utterance_id": f"{prefix}_{i:05d}",
                    "text": "".join(m.get("surface", "") for m in morphemes),
                    "morphemes": morphemes,
                }
            )
    return generated


def _generate_phrase_replaced(
    phrase_groups: list[list[list[dict]]],
    all_phrases: list[list[dict]],
    num: int,
    rng: random.Random,
) -> list[dict]:
    """単一句置換.

    Returns:
        生成された発話リスト.

    """
    phrases_by_len: dict[int, list[list[dict]]] = {}
    for p in all_phrases:
        plen = len(p)
        if plen not in phrases_by_len:
            phrases_by_len[plen] = []
        phrases_by_len[plen].append(p)

    generated = []
    for i in range(num):
        group = rng.choice(phrase_groups)
        replace_idx = rng.randint(0, len(group) - 1)
        original_phrase = group[replace_idx]
        orig_len = len(original_phrase)

        candidates = []
        for delta in [0, -1, 1, -2, 2]:
            target_len = orig_len + delta
            if target_len in phrases_by_len:
                candidates.extend(phrases_by_len[target_len])
            if len(candidates) >= 10:
                break

        if not candidates:
            continue

        replacement = rng.choice(candidates)
        morphemes = []
        for j, phrase in enumerate(group):
            if j == replace_idx:
                morphemes.extend(replacement)
            else:
                morphemes.extend(phrase)

        if morphemes:
            generated.append(
                {
                    "utterance_id": f"REPLACE_{i:05d}",
                    "text": "".join(m.get("surface", "") for m in morphemes),
                    "morphemes": morphemes,
                }
            )
    return generated


def main() -> None:
    """エントリポイント."""
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Generate v23 combined corpus")
    parser.add_argument(
        "--jsut",
        type=Path,
        default=Path(os.environ.get("FINETUNE_DATA", "")),
    )
    parser.add_argument(
        "--original-corpus",
        type=Path,
        default=Path(os.environ.get("CORPUS_DATA", "")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "combined_corpus_v3.json",
    )
    args = parser.parse_args()

    # Load JSUT (training split only)
    print(f"Loading JSUT: {args.jsut}")
    with open(args.jsut, encoding="utf-8") as f:
        jsut_data = json.load(f)
    utterances = jsut_data["utterances"]
    random.seed(42)
    indices = list(range(len(utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(utterances) if i not in val_idx]
    print(f"  Training split: {len(train_utts)} utterances")

    phrases = _extract_accent_phrases(train_utts)
    phrase_groups = _extract_phrase_groups(train_utts)
    print(f"  Accent phrases: {len(phrases)}")

    combined: list[dict] = []

    # 1. Original LLM corpus (420)
    if args.original_corpus.exists():
        print(f"\nLoading LLM corpus: {args.original_corpus}")
        with open(args.original_corpus, encoding="utf-8") as f:
            orig_data = json.load(f)
        if isinstance(orig_data, list):
            orig_utts = orig_data
        else:
            orig_utts = orig_data.get("utterances", [])
        print(f"  {len(orig_utts)} utterances")
        combined.extend(orig_utts)

    # 2. Recombined: seed=42, 3-6 phrases (longer than v20's 2-5)
    rng1 = random.Random(42)
    print("\nGenerating recombined (seed=42, 3-6 phrases): 1200")
    recomb1 = _generate_recombined(phrases, 1200, 3, 6, rng1, "RECOMB_A")
    print(f"  {len(recomb1)} utterances")
    combined.extend(recomb1)

    # 3. Recombined: seed=77, 3-6 phrases (different combos for diversity)
    rng2 = random.Random(77)
    print("Generating recombined (seed=77, 3-6 phrases): 800")
    recomb2 = _generate_recombined(phrases, 800, 3, 6, rng2, "RECOMB_B")
    print(f"  {len(recomb2)} utterances")
    combined.extend(recomb2)

    # 4. Phrase-replaced (seed=42)
    rng3 = random.Random(42)
    print("\nGenerating phrase-replaced: 1000")
    replaced = _generate_phrase_replaced(phrase_groups, phrases, 1000, rng3)
    print(f"  {len(replaced)} utterances")
    combined.extend(replaced)

    # Stats
    total = len(combined)
    total_morphemes = sum(len(u.get("morphemes", [])) for u in combined)
    avg_len = total_morphemes / total if total else 0
    print("\n=== Combined corpus v3 ===")
    print(f"  Total: {total} utterances")
    print(f"  Avg morphemes/utterance: {avg_len:.1f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
