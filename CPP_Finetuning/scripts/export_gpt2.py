"""
export_gpt2.py
--------------
One-time bootstrap script (run in Python).
Downloads GPT-2 small from HuggingFace and produces:
  1. gpt2_weights.pt   – named-parameter archive loadable in C++ / LibTorch
  2. vocab.json        – BPE vocabulary (token -> id)
  3. merges.txt        – BPE merge rules

Usage:
    pip install transformers torch
    python scripts/export_gpt2.py --out-dir .
"""

import argparse
import os
import shutil
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def export(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading GPT-2 small …")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # ── 1. Save named parameters as a TorchScript archive ─────────────────────
    # We script a thin wrapper so that torch::jit::load() in C++ can iterate
    # named_parameters() and copy them into the native C++ model.
    class WeightHolder(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            for k, v in state_dict.items():
                # Replace dots with double-underscore so the name is JIT-safe
                safe_key = k.replace(".", "__")
                self.register_parameter(safe_key, torch.nn.Parameter(v, requires_grad=False))

    holder = torch.jit.script(WeightHolder(model.state_dict()))
    weight_path = os.path.join(out_dir, "gpt2_weights.pt")
    holder.save(weight_path)
    print(f"  → {weight_path}  ({os.path.getsize(weight_path) / 1e6:.1f} MB)")

    # ── 2. Save vocab / merges for the C++ BPE tokenizer ──────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    cache_dir = tokenizer.save_pretrained("/tmp/gpt2_tokenizer")

    for fname in ("vocab.json", "merges.txt"):
        src = os.path.join("/tmp/gpt2_tokenizer", fname)
        dst = os.path.join(out_dir, fname)
        shutil.copy(src, dst)
        print(f"  → {dst}")

    print("\nDone! Place gpt2_weights.pt, vocab.json and merges.txt in the project root.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()
    export(args.out_dir)
