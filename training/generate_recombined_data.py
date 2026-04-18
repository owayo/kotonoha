"""JSUT アクセント句リコンビネーションによる合成データ生成.

JSUT の発話からアクセント句（同じ accent_type を持つ連続形態素群）を抽出し、
異なる発話の句を組み合わせて新しい発話を生成する。

各アクセント句のラベルは元の発話から正確に保持されるため高品質。
生成データは corpus_converted.json と同じ形式で出力。

Usage:
    python generate_recombined_data.py --input JSUT --output OUT [--num 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _extract_accent_phrases(utterances: list[dict]) -> list[list[dict]]:
    """発話リストからアクセント句を抽出する.

    Returns:
        アクセント句のリスト（各句は形態素リスト）.

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


def _generate_recombined_utterances(
    phrases: list[list[dict]],
    num_utterances: int = 2000,
    min_phrases: int = 2,
    max_phrases: int = 5,
    seed: int = 42,
) -> list[dict]:
    """アクセント句をランダムに組み合わせて新しい発話を生成する.

    Returns:
        生成された発話のリスト.

    """
    rng = random.Random(seed)
    generated = []

    for i in range(num_utterances):
        n_phrases = rng.randint(min_phrases, max_phrases)
        selected = rng.sample(phrases, min(n_phrases, len(phrases)))

        # Concatenate morphemes from selected phrases
        morphemes = []
        for phrase in selected:
            morphemes.extend(phrase)

        if not morphemes:
            continue

        generated.append(
            {
                "utterance_id": f"RECOMB_{i:05d}",
                "text": "".join(m.get("surface", "") for m in morphemes),
                "morphemes": morphemes,
            }
        )

    return generated


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


def main() -> None:
    """エントリポイント."""
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate recombined accent data from JSUT"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.environ.get("FINETUNE_DATA", "")),
        help="JSUT data path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "recombined_corpus.json",
        help="Output path",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=2000,
        help="Number of utterances to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-phrases", type=int, default=2)
    parser.add_argument("--max-phrases", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading JSUT: {args.input}")
    with open(args.input, encoding="utf-8") as f:
        jsut_data = json.load(f)

    utterances = jsut_data["utterances"]
    print(f"  {len(utterances)} utterances")

    # Use only training split (same as training scripts: seed=42, 90/10)
    random.seed(42)
    indices = list(range(len(utterances)))
    random.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    val_idx = set(indices[:val_size])
    train_utts = [u for i, u in enumerate(utterances) if i not in val_idx]
    print(f"  Training split: {len(train_utts)} utterances")

    # Extract accent phrases from training data only
    phrases = _extract_accent_phrases(train_utts)
    print(f"  Extracted {len(phrases)} accent phrases")

    # Generate recombined utterances
    generated = _generate_recombined_utterances(
        phrases,
        num_utterances=args.num,
        min_phrases=args.min_phrases,
        max_phrases=args.max_phrases,
        seed=args.seed,
    )
    print(f"  Generated {len(generated)} recombined utterances")

    # Stats
    total_morphemes = sum(len(u["morphemes"]) for u in generated)
    avg_len = total_morphemes / len(generated) if generated else 0
    print(f"  Avg morphemes/utterance: {avg_len:.1f}")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(generated, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
