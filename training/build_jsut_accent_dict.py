"""JSUT v3 + corpus から dict_accent_type を抽出して accent_dict_jsut.csv を作る.

JSUT のデータ準備パイプライン (`text_to_accent_json.py`) は accent_dict.csv +
accent_dict_kanjium.csv に加えて corpus_lookup と UniDic aType を併用する。
本番 Rust ランタイムは UniDic を持たないため、JSUT 由来の (lemma, reading) →
dict_accent_type マップを CSV にエクスポートして同梱することで lookup-only
モードでも JSUT と同等の dict_accent_type を再現する。

出力:
- /mnt/c/GitHub/kotonoha-models/accent_dict_jsut.csv (既存 dict と同形式)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_onnx_v60 import _enrich_utterances, _load_accent_dicts, _load_dotenv


def main() -> None:
    """Extract dict_accent_type values from JSUT + corpus and write CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/mnt/c/GitHub/kotonoha-models/accent_dict_jsut.csv",
    )
    parser.add_argument(
        "--jsut",
        default="/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json",
    )
    parser.add_argument(
        "--corpus",
        default="/mnt/c/GitHub/kotonoha-training-data/train/corpus_converted.json",
    )
    args = parser.parse_args()

    _load_dotenv()
    # 訓練側 `_enrich_utterances` と同じ 2 dict をマージしてロードし、JSUT を
    # 一旦 enrich してから dict_accent_type を吸い出す。これで「post-enrich」値、
    # すなわち実際に v66_split1 訓練に入った値を抽出できる。
    enrich_dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    enrich_dict = _load_accent_dicts(enrich_dict_paths)

    seen: dict[tuple[str, str], str] = {}

    def add_morpheme(m: dict) -> None:
        lemma = str(m.get("lemma", "")).strip()
        reading = str(m.get("reading", "")).strip()
        if not lemma or not reading:
            return
        dat = m.get("dict_accent_type", "*")
        if dat in ("*", "", None):
            return
        s = str(dat).strip().strip('"')
        if not s.lstrip("-").isdigit():
            return
        try:
            n = int(s)
        except ValueError:
            return
        if n < 0:
            return
        # `(lemma, reading)` キーで上書き優先 (後勝ち)。Rust の enrich は
        # `_enrich_utterances` と同じ (lemma, reading) + dash fallback のみを
        # 使うので、追加 surface/pronunciation キーは含めない。
        seen[(lemma, reading)] = str(n)

    # JSUT (enrich first to get post-enrich dict_accent_type)
    with open(args.jsut, encoding="utf-8") as f:
        jsut = json.load(f).get("utterances", [])
    _enrich_utterances(jsut, enrich_dict)
    for utt in jsut:
        for m in utt.get("morphemes", []):
            add_morpheme(m)
    print(f"After JSUT: {len(seen)} entries")

    # corpus_converted (enrich too)
    corpus_path = Path(args.corpus)
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            data = json.load(f)
        utterances = data if isinstance(data, list) else data.get("utterances", [])
        _enrich_utterances(utterances, enrich_dict)
        for utt in utterances:
            for m in utt.get("morphemes", []):
                add_morpheme(m)
        print(f"After corpus: {len(seen)} entries")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("lemma,reading,accent_type\n")
        for (lemma, reading), acc in sorted(seen.items()):
            # CSV-safe: 値にカンマ等が含まれないことを期待
            f.write(f"{lemma},{reading},{acc}\n")
    print(f"Wrote {len(seen)} entries to {out}")


if __name__ == "__main__":
    main()
