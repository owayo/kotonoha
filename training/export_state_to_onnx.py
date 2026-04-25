"""Export a saved state.pt to ONNX (for v38-style 14-dim models)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from train_onnx_v38 import NUM_CLASSES, FEATURE_DIM, AccentModel, _OnnxWrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bundle = torch.load(args.state, map_location="cpu", weights_only=False)
    if "state" in bundle:
        state = bundle["state"]
    else:
        state = bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentModel(
        embed_dim=64,
        hidden_dim=256,
        num_layers=3,
        num_classes=NUM_CLASSES,
        dropout=0.4,
        attention_heads=4,
        reading_dropout=0.0,
    ).to(device)
    to_load = {}
    for k, v in model.state_dict().items():
        if k in state:
            to_load[k] = state[k].to(dtype=v.dtype, device=device)
        else:
            to_load[k] = v
    model.load_state_dict(to_load)
    model.eval()

    wrapper = _OnnxWrapper(model).to(device)
    wrapper.eval()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(10, FEATURE_DIM, device=device)
    dummy[:, 0] = 1
    dummy[:, 1] = 2
    dummy[:, 5] = 0.3
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "seq_len"}, "output": {0: "seq_len"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {out_path}")


if __name__ == "__main__":
    main()
