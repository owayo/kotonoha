"""v57: Generate high-confidence pseudo-labels for JVS using v38+v56 ensemble.

For each JVS utterance:
  - Run v38 ensemble (v38.onnx + v56 5 seeds)
  - At each morpheme position, compute max softmax prob (confidence)
  - If confidence > threshold (default 0.95), accept the prediction
  - Replace accent_type with predicted value
  - Save filtered JVS for v57 training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from train_onnx_v38 import (
    NUM_CLASSES,
    AccentModel,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
)


def _softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=-1, keepdims=True)


def main() -> None:
    """Generate confidence-filtered pseudo-labeled JVS."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument(
        "--jvs",
        default="/home/owayo/kotonoha-training/data/jvs_accent_data.json",
    )
    ap.add_argument(
        "--output",
        default="/mnt/c/GitHub/kotonoha-training-data/train/jvs_pseudo_v57.json",
    )
    args = ap.parse_args()

    dict_paths = [
        Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
        Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
    ]
    accent_dict = _load_accent_dicts(dict_paths)
    with open(args.jvs, encoding="utf-8") as f:
        jvs_data = json.load(f)
    jvs_utts = jvs_data["utterances"]
    _enrich_utterances(jvs_utts, accent_dict)
    print(f"JVS: {len(jvs_utts)} utts")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_v24 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers
    )
    sess_v38 = ort.InferenceSession(
        "/mnt/c/GitHub/kotonoha-models/accent_model_v38.onnx", providers=providers
    )

    # Load v56 5 seeds (val_split=0 trained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v56_models = []
    for i in range(5):
        sf = Path(f"/tmp/v56_states/state_{i:03d}.pt")
        if sf.exists():
            bundle = torch.load(sf, map_location="cpu", weights_only=False)
            state = bundle["state"]
            m = AccentModel(
                embed_dim=64,
                hidden_dim=256,
                num_layers=3,
                num_classes=NUM_CLASSES,
                dropout=0.4,
                attention_heads=4,
                reading_dropout=0.0,
            ).to(device)
            to_load = {}
            for k, v in m.state_dict().items():
                if k in state:
                    to_load[k] = state[k].to(dtype=v.dtype, device=device)
                else:
                    to_load[k] = v
            m.load_state_dict(to_load)
            m.eval()
            v56_models.append((f"v56s{i}", m))
    print(f"v56 models loaded: {len(v56_models)}")

    pseudo_utts = []
    accept_total = 0
    accept_match = 0  # pseudo == orig label
    morph_total = 0

    with torch.no_grad():
        for utt_idx, utt in enumerate(jvs_utts):
            ms = utt.get("morphemes", [])
            if not ms:
                continue
            n = len(ms)
            f13 = np.array(
                [
                    _extract_morpheme_features(m, j / max(n - 1, 1))
                    for j, m in enumerate(ms)
                ],
                dtype=np.float32,
            )
            # v38 needs 14 dim with v24 argmax
            v24_log = sess_v24.run(None, {"input": f13[:, :11]})[0]
            v24_arg = v24_log.argmax(-1)
            f14 = np.concatenate(
                [
                    f13[:, :13],
                    (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1),
                ],
                axis=1,
            )
            v38_sm = _softmax(sess_v38.run(None, {"input": f14})[0])
            stacked = [v38_sm]
            # v56 models (also 14 dim input)
            features = torch.tensor(f14, device=device).unsqueeze(0)
            lengths = torch.tensor([n])
            r_ids = torch.zeros((1, n, 12), dtype=torch.long, device=device)
            for _name, m in v56_models:
                logits = m(features, lengths, r_ids)
                sm = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                stacked.append(sm)
            avg = np.mean(stacked, axis=0)
            preds = avg.argmax(-1)
            confs = avg.max(-1)

            new_morphs = []
            for i_m, morph in enumerate(ms):
                m2 = dict(morph)
                if confs[i_m] >= args.threshold:
                    new_label = int(preds[i_m])
                    orig_label = morph.get("accent_type", 0)
                    if new_label == orig_label:
                        accept_match += 1
                    m2["accent_type"] = new_label
                    accept_total += 1
                morph_total += 1
                new_morphs.append(m2)
            pseudo_utts.append({"morphemes": new_morphs})

    print(f"\nThreshold: {args.threshold}")
    print(f"Total morphemes: {morph_total}")
    print(
        f"Accepted (conf >= {args.threshold}): {accept_total} ({accept_total / morph_total * 100:.2f}%)"
    )
    if accept_total > 0:
        print(
            f"Pseudo == orig label: {accept_match}/{accept_total} ({accept_match / accept_total * 100:.2f}%)"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pseudo_utts, f, ensure_ascii=False)
    print(f"Saved {len(pseudo_utts)} utts to {out_path}")


if __name__ == "__main__":
    main()
