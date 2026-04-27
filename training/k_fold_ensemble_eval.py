"""K-fold ensemble evaluation across val_splits.

Models:
  - v38 (val_split=0 で学習, leak なし on val_split=0)
  - v54_split1 (val_split=1 で学習, leak ありのため val_split=0 上で 86.47%)
  - v54_split2 (val_split=2 で学習)
  - v54_split3 (val_split=3 で学習)

各 split N で eval した時の各 model の精度と、ensemble 効果を測定。
val_split=N で eval する際、v54_splitN は完全 in-train、他は val_split=N の
utts を ~90% train で見ている。Disjoint K-fold ではないので、leak ありの参考値。

Output:
  各 split N について:
    - 単独精度 (4 model 各)
    - 全 4 model ensemble 精度
    - val_split=N model を除外した 3 model ensemble (OOF 風)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from train_onnx_v38 import (
    NUM_CLASSES,
    _enrich_utterances,
    _extract_morpheme_features,
    _load_accent_dicts,
    _load_dotenv,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx)
    return e / np.sum(e, axis=-1, keepdims=True)


def _split_val(jsut: list, val_split_seed: int) -> set[int]:
    indices = list(range(len(jsut)))
    rng = random.Random(val_split_seed)
    rng.shuffle(indices)
    val_size = int(len(indices) * 0.1)
    return set(indices[:val_size])


def _compute_softmax_for_utts(
    utts: list,
    sess_v24: ort.InferenceSession,
    sessions_14d: dict[str, ort.InferenceSession],
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {n: [] for n in sessions_14d}
    out["v24"] = []
    all_labels: list[np.ndarray] = []
    for utt in utts:
        ms = utt.get("morphemes", [])
        if not ms:
            continue
        n = len(ms)
        feats13 = np.array(
            [
                _extract_morpheme_features(m, i / max(n - 1, 1))
                for i, m in enumerate(ms)
            ],
            dtype=np.float32,
        )
        labels = np.array(
            [min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms],
            dtype=np.int64,
        )
        all_labels.append(labels)

        v24_log = sess_v24.run(None, {"input": feats13[:, :11]})[0]
        out["v24"].append(_softmax(v24_log))
        v24_arg = v24_log.argmax(-1)
        feats14 = np.concatenate(
            [feats13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)],
            axis=1,
        )
        for name, sess in sessions_14d.items():
            log = sess.run(None, {"input": feats14})[0]
            out[name].append(_softmax(log))
    return out, all_labels


def _accuracy(
    preds_per_utt: list[np.ndarray], labels_per_utt: list[np.ndarray]
) -> float:
    flat_p = np.concatenate(preds_per_utt)
    flat_l = np.concatenate(labels_per_utt)
    return float((flat_p == flat_l).mean())


def main() -> None:
    """Run K-fold partial-leak ensemble eval across 4 val_splits."""
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

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    models_dir = Path("/mnt/c/GitHub/kotonoha-models")
    sess_v24 = ort.InferenceSession(
        str(models_dir / "accent_model_v24.onnx"), providers=providers
    )
    sessions_14d = {
        "v38": ort.InferenceSession(
            str(models_dir / "accent_model_v38.onnx"), providers=providers
        ),
        "v54_s1": ort.InferenceSession(
            str(models_dir / "accent_model_v54_split1.onnx"), providers=providers
        ),
        "v54_s2": ort.InferenceSession(
            str(models_dir / "accent_model_v54_split2.onnx"), providers=providers
        ),
        "v54_s3": ort.InferenceSession(
            str(models_dir / "accent_model_v54_split3.onnx"), providers=providers
        ),
    }
    print(f"Loaded models: v24, {list(sessions_14d.keys())}")

    # split N の owner model: 該当 split 学習 model
    owner = {0: "v38", 1: "v54_s1", 2: "v54_s2", 3: "v54_s3"}

    for split_n in [0, 1, 2, 3]:
        val_idx = _split_val(jsut, split_n)
        val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
        print(f"\n=== val_split={split_n}: {len(val_utts)} utts ===")
        smx, labels = _compute_softmax_for_utts(val_utts, sess_v24, sessions_14d)
        total_morph = sum(len(la) for la in labels)
        print(f"morphemes: {total_morph}")

        # Single
        for name in ["v24"] + list(sessions_14d.keys()):
            preds = [sm.argmax(-1) for sm in smx[name]]
            acc = _accuracy(preds, labels)
            mark = (
                " (OOF)"
                if name == owner.get(split_n) or name == "v24"
                else " (in-train, leak)"
            )
            print(f"  {name}: {acc * 100:.2f}%{mark}")

        # All 4 ensemble (v24 + 3 leak + 1 OOF), and 4-of-14d ensemble
        names_14 = list(sessions_14d.keys())
        # 4-model ensemble (all v54_s*+v38)
        avg_preds = []
        for i in range(len(labels)):
            stacked = np.stack([smx[n][i] for n in names_14])
            avg_preds.append(stacked.mean(0).argmax(-1))
        acc_all = _accuracy(avg_preds, labels)
        print(f"  ensemble[v38+v54_s1+v54_s2+v54_s3]: {acc_all * 100:.2f}%")

        # OOF-like: 該当 split owner を除外した 3-model ensemble
        owner_n = owner[split_n]
        excl_names = [n for n in names_14 if n != owner_n]
        avg_preds_excl = []
        for i in range(len(labels)):
            stacked = np.stack([smx[n][i] for n in excl_names])
            avg_preds_excl.append(stacked.mean(0).argmax(-1))
        acc_excl = _accuracy(avg_preds_excl, labels)
        excl_tag = f"3-model, exclude {owner_n}, all leak"
        print(f"  ensemble[{excl_tag}]: {acc_excl * 100:.2f}%")

        # Owner model のみ vs ensemble の比較
        owner_preds = [sm.argmax(-1) for sm in smx[owner_n]]
        acc_owner = _accuracy(owner_preds, labels)
        print(f"  owner[{owner_n}] only: {acc_owner * 100:.2f}%")

        # Weighted: owner に高い weight、他 leak model に低い weight
        for owner_w in [0.5, 0.6, 0.7]:
            other_w = (1.0 - owner_w) / 3.0
            avg_preds_w = []
            for i in range(len(labels)):
                stacked = []
                weights = []
                for n in names_14:
                    stacked.append(smx[n][i])
                    weights.append(owner_w if n == owner_n else other_w)
                stacked_arr = np.stack(stacked)  # [4, seq, 21]
                w_arr = np.array(weights, dtype=np.float32).reshape(-1, 1, 1)
                avg = (stacked_arr * w_arr).sum(0)
                avg_preds_w.append(avg.argmax(-1))
            acc_w = _accuracy(avg_preds_w, labels)
            print(f"  weighted[owner={owner_w:.1f}]: {acc_w * 100:.2f}%")

    # Cross-split summary: OOF 集計 (全データ精度推定)
    print("\n=== Cross-split OOF summary (各 split で owner model のみ) ===")
    total_correct = 0
    total_morph = 0
    for split_n in [0, 1, 2, 3]:
        val_idx = _split_val(jsut, split_n)
        val_utts = [u for i, u in enumerate(jsut) if i in val_idx]
        owner_n = owner[split_n]
        smx, labels = _compute_softmax_for_utts(
            val_utts, sess_v24, {owner_n: sessions_14d[owner_n]}
        )
        preds = [sm.argmax(-1) for sm in smx[owner_n]]
        acc = _accuracy(preds, labels)
        n_correct = sum((p == la).sum() for p, la in zip(preds, labels, strict=True))
        n_total = sum(len(la) for la in labels)
        total_correct += n_correct
        total_morph += n_total
        owner_tag = f"split={split_n} owner={owner_n}"
        print(f"  {owner_tag}: {acc * 100:.2f}% ({n_correct}/{n_total})")
    oof_acc = total_correct / total_morph * 100
    print(f"  OOF aggregate (no leak): {oof_acc:.2f}% ({total_correct}/{total_morph})")


if __name__ == "__main__":
    main()
