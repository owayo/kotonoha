"""Quick: evaluate v48 single + add to ensemble."""

import json, random
from pathlib import Path
import numpy as np
import onnxruntime as ort
from train_onnx_v38 import NUM_CLASSES, _enrich_utterances, _extract_morpheme_features, _load_accent_dicts


def _softmax(x):
    mx = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - mx); return e / np.sum(e, axis=-1, keepdims=True)


dict_paths = [
    Path("/mnt/c/GitHub/kotonoha/data/accent_dict.csv"),
    Path("/mnt/c/GitHub/kotonoha-training-data/train/accent_dict.csv"),
]
ad = _load_accent_dicts(dict_paths)
with open("/home/owayo/kotonoha-training/data/jsut_accent_data_v3.json", encoding="utf-8") as f:
    jsut = json.load(f)["utterances"]
_enrich_utterances(jsut, ad)
random.seed(0)
idx = list(range(len(jsut))); random.shuffle(idx)
val_size = int(len(idx) * 0.1); val_idx = set(idx[:val_size])
val_utts = [u for i, u in enumerate(jsut) if i in val_idx]

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess_v24 = ort.InferenceSession("/mnt/c/GitHub/kotonoha-models/accent_model_v24.onnx", providers=providers)
sess_v20 = ort.InferenceSession("/mnt/c/GitHub/kotonoha-models/accent_model_v20.onnx", providers=providers)
sess_v17 = ort.InferenceSession("/mnt/c/GitHub/kotonoha-models/accent_model_v17.onnx", providers=providers)
sess_v48 = ort.InferenceSession("/mnt/c/GitHub/kotonoha-models/accent_model_v48.onnx", providers=providers)

s_v20, s_v17, s_v48, s_v20_tta, labels = [], [], [], [], []
rng = np.random.default_rng(0)
for utt in val_utts:
    ms = utt.get("morphemes", [])
    if not ms: continue
    n = len(ms)
    f13 = np.array([_extract_morpheme_features(m, i / max(n - 1, 1)) for i, m in enumerate(ms)], dtype=np.float32)
    labs = np.array([min(m.get("accent_type", 0), NUM_CLASSES - 1) for m in ms], dtype=np.int64)
    labels.append(labs)
    v24_arg = sess_v24.run(None, {"input": f13[:, :11]})[0].argmax(-1)
    f14 = np.concatenate([f13[:, :13], (v24_arg.astype(np.float32) / 20.0).reshape(-1, 1)], axis=1)
    s_v20.append(_softmax(sess_v20.run(None, {"input": f13[:, :11]})[0]))
    s_v17.append(_softmax(sess_v17.run(None, {"input": f13[:, :11]})[0]))
    s_v48.append(_softmax(sess_v48.run(None, {"input": f14})[0]))
    # v20 TTA
    sm = _softmax(sess_v20.run(None, {"input": f13[:, :11]})[0])
    for _ in range(8):
        aug = f13[:, :11].copy()
        aug[:, 5:] += rng.normal(0, 0.02, aug[:, 5:].shape).astype(np.float32)
        sm = sm + _softmax(sess_v20.run(None, {"input": aug})[0])
    s_v20_tta.append(sm / 9)

flat = np.concatenate(labels)

def acc_combo(parts):
    preds = []
    for i in range(len(labels)):
        avg = sum(w * sm[i] for w, sm in parts)
        preds.append(avg.argmax(-1))
    return (np.concatenate(preds) == flat).mean()

# Singles
print(f"v20: {acc_combo([(1, s_v20)])*100:.2f}%")
print(f"v17: {acc_combo([(1, s_v17)])*100:.2f}%")
print(f"v48: {acc_combo([(1, s_v48)])*100:.2f}%")
print(f"v20_tta: {acc_combo([(1, s_v20_tta)])*100:.2f}%")
print(f"v20+v17: {acc_combo([(0.5, s_v20), (0.5, s_v17)])*100:.2f}%")
print(f"v20+v17+v48: {acc_combo([(0.33, s_v20), (0.33, s_v17), (0.33, s_v48)])*100:.2f}%")
print(f"v20+v17_tta+v48: {acc_combo([(0.4, s_v20), (0.3, s_v17), (0.3, s_v48)])*100:.2f}%")
# Random search 4 members
all_sm = [("v20", s_v20), ("v17", s_v17), ("v48", s_v48), ("v20_tta", s_v20_tta)]
best = 0; best_w = None
for _ in range(2000):
    w = rng.dirichlet(np.ones(4))
    parts = [(w[k], all_sm[k][1]) for k in range(4)]
    a = acc_combo(parts)
    if a > best: best = a; best_w = w
print(f"\nBest 4-way weighted: {best*100:.2f}%")
print(f"  weights: " + " ".join(f"{n}={w:.2f}" for (n, _), w in zip(all_sm, best_w)))
