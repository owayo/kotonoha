"""Python ↔ Rust feat13 数値一致テスト用 fixture を生成する.

JSUT 50 utts の各形態素について:
- 入力フィールド (surface, pos, ..., reading) を JSON 化
- post-enrich `dict_accent_type` (string, "0".."20" or "*")
- post-enrich `position` (i / max(n-1, 1))
- 期待値 `feat13` ([13] float32)

Rust 側 `tests/v66_feat13_match.rs` で読み込み、`extract_feat13` の結果と
完全一致すること (1e-7 以内) を確認する。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from train_onnx_v60 import (
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
    _load_dotenv,
)


def main() -> None:
    """Build feat13 fixture for the first N JSUT utts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/mnt/c/GitHub/kotonoha/kotonoha/tests/fixtures/v66_feat13.json",
    )
    parser.add_argument("--num-utts", type=int, default=50)
    parser.add_argument("--val-split-seed", type=int, default=0)
    args = parser.parse_args()

    _load_dotenv()
    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)
    jsut_path = "/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json"
    with open(jsut_path, encoding="utf-8") as f:
        jsut = json.load(f)["utterances"]
    _enrich_utterances(jsut, accent_dict)

    # 再現性のため固定 seed で抽出
    rng = random.Random(args.val_split_seed)
    indices = list(range(len(jsut)))
    rng.shuffle(indices)
    test_idx = sorted(indices[: args.num_utts])
    selected = [jsut[i] for i in test_idx]

    fixture: list[dict] = []
    for utt in selected:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        n = len(ms)
        feats = []
        for i, m in enumerate(ms):
            position = i / max(n - 1, 1)
            feats.append(_extract_morpheme_features(m, position))
        morphemes_dump = []
        for m in ms:
            morphemes_dump.append(
                {
                    "surface": m.get("surface", ""),
                    "pos": m.get("pos", ""),
                    "pos_detail1": m.get("pos_detail1", "*"),
                    "pos_detail2": m.get("pos_detail2", "*"),
                    "conjugation_type": m.get("conjugation_type", "*"),
                    "conjugation_form": m.get("conjugation_form", "*"),
                    "lemma": m.get("lemma", ""),
                    "reading": m.get("reading", ""),
                    "pronunciation": m.get("pronunciation", ""),
                    "dict_accent_type": m.get("dict_accent_type", "*"),
                }
            )
        fixture.append(
            {
                "utterance_id": utt.get("utterance_id", ""),
                "morphemes": morphemes_dump,
                "feat13": feats,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"utterances": fixture}, f, ensure_ascii=False, indent=None)
    print(f"Wrote {len(fixture)} utts to {out}")


if __name__ == "__main__":
    main()
