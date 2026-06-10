# GPT-2 LoRA Fine-Tuning in C++ with LibTorch

Fine-tune GPT-2 (124M) from scratch in pure C++ using LibTorch and LoRA adapters. Designed for an RTX 4060 (8 GB VRAM) — LoRA keeps VRAM usage under 2 GB in FP32.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| LibTorch (CUDA) | Download from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) — select **C++ / Java**, CUDA 12.x |
| CMake ≥ 3.18 | `winget install Kitware.CMake` |
| MSVC 2022 | Visual Studio Build Tools |
| Python + HuggingFace | For the one-time bootstrap only |
| nlohmann/json | Fetched automatically by CMake |

---

## Quick Start

### Step 1 — Bootstrap (one-time, Python)

```bash
pip install transformers torch
python scripts/export_gpt2.py --out-dir .
# Produces: gpt2_weights.pt  vocab.json  merges.txt
```

### Step 2 — Build

```powershell
cd CPP_Finetuning
mkdir build; cd build

# Replace C:/libtorch with your actual LibTorch path
cmake .. -DCMAKE_PREFIX_PATH="C:/libtorch" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Step 3 — Verify Forward Pass

```powershell
.\Release\finetuner.exe --mode test_forward --weights ..\gpt2_weights.pt --vocab ..\vocab.json --merges ..\merges.txt
# Expected: Output shape: [1, 64, 50257]
```

### Step 4 — Fine-Tune

```powershell
.\Release\finetuner.exe `
  --mode      train `
  --weights   ..\gpt2_weights.pt `
  --vocab     ..\vocab.json `
  --merges    ..\merges.txt `
  --data      ..\data\sample.txt `
  --steps     1000 `
  --batch-size 4 `
  --block-size 512 `
  --log-every 10 `
  --save-every 200
```

### Step 5 — Generate Text

```powershell
.\Release\finetuner.exe `
  --mode      generate `
  --weights   ..\gpt2_weights.pt `
  --vocab     ..\vocab.json `
  --merges    ..\merges.txt `
  --lora      ..\checkpoints\lora_step1000.pt `
  --prompt    "The optimization problem is" `
  --max-tokens 200
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `test_forward`, or `generate` |
| `--weights` | `gpt2_weights.pt` | Pretrained weight archive |
| `--vocab` | `vocab.json` | BPE vocabulary |
| `--merges` | `merges.txt` | BPE merge rules |
| `--data` | `data/sample.txt` | Training text file |
| `--lora` | *(empty)* | LoRA checkpoint to resume from |
| `--prompt` | `"Once upon a time"` | Prompt for generate mode |
| `--steps` | `1000` | Total training steps |
| `--batch-size` | `4` | Sequences per mini-batch |
| `--block-size` | `512` | Context window (tokens) |
| `--log-every` | `10` | Steps between loss logs |
| `--save-every` | `200` | Steps between checkpoints |
| `--max-tokens` | `200` | Max tokens to generate |
| `--no-cuda` | *(off)* | Force CPU execution |

---

## Project Structure

```
CPP_Finetuning/
├── CMakeLists.txt
├── scripts/
│   └── export_gpt2.py       # One-time: exports weights + vocab
├── include/
│   ├── config.h             # GPT2Config, LoRAConfig
│   ├── model/gpt2.h         # Native GPT-2 architecture
│   ├── lora/lora_linear.h   # LoRALinear module
│   ├── tokenizer/bpe_tokenizer.h
│   ├── data/dataset.h
│   └── training/trainer.h
├── src/
│   ├── main.cpp
│   ├── model/gpt2.cpp
│   ├── lora/lora_linear.cpp
│   ├── tokenizer/bpe_tokenizer.cpp
│   ├── data/dataset.cpp
│   └── training/trainer.cpp
├── data/
│   └── sample.txt           # Put your training text here
└── checkpoints/             # LoRA adapter weights saved here
```

---

## LoRA Details

| Parameter | Value |
|---|---|
| Rank (`r`) | 8 |
| Alpha (`α`) | 16 |
| Scaling | `α / r = 2.0` |
| Target layers | Attention `c_attn` (Q, K, V) in all 12 blocks |
| Trainable params | ~614 K / 124 M total (0.5%) |
| VRAM (FP32) | ~2 GB on RTX 4060 at batch=4, block=512 |

---

## Adding Your Own Dataset

Place a plain UTF-8 `.txt` file in `data/` and point `--data` at it. The dataset loader tokenizes the full text and splits it into non-overlapping `block_size`-token windows. A minimum of ~50 K tokens is recommended for training to converge meaningfully.
