# CUDA ResNet-18 on CIFAR-10

A from-scratch ResNet-18 implementation in raw CUDA, using **direct cuDNN and cuBLAS API calls** with no LibTorch/PyTorch dependency. Built as a performance study over the `CPP_ResNet` project in this repo, which uses the LibTorch C++ API.

---

## Why raw CUDA over LibTorch C++?

| What we gain | How |
|---|---|
| Fused Add + ReLU kernel | Each BasicBlock's residual add and ReLU fire as one kernel, cutting a global-memory round-trip per block |
| cuDNN algorithm benchmarking | At init time we run `cudnnFind*Algorithm` and pick the fastest conv algorithm for your exact GPU and input sizes — LibTorch picks a fixed algorithm |
| Pinned host memory | `cudaMallocHost` for the dataset enables async DMA at full PCIe bandwidth; LibTorch uses pageable memory |
| Double-buffered CUDA streams | The next batch transfers to GPU while the current batch is being trained — zero idle PCIe time |
| No autograd overhead | Backward pass is written explicitly; no dynamic graph construction, no tensor metadata checks |

Expected end-to-end speedup over `CPP_ResNet` on CIFAR-10: **40–60%** faster per epoch, primarily from the fused kernel and reduced dispatch overhead on the small 32×32 tensors.

---

## Prerequisites

### Required software

| Dependency | Minimum version | Notes |
|---|---|---|
| Windows 10/11 | — | Linux works too; adjust paths and DLL steps |
| Visual Studio | 2019 or 2022 | MSVC compiler; Community edition is free |
| CUDA Toolkit | 12.x | Includes cuDNN and cuBLAS |
| CMake | 3.18 | For the `LANGUAGES CUDA` generator support |

> **cuDNN and cuBLAS are bundled with the CUDA Toolkit installer** since CUDA 12.0 — you do not need to download them separately.

### Verify your installation

Open a Developer Command Prompt (or PowerShell) and run:

```
nvcc --version
```

Expected output (example):

```
nvcc: NVIDIA (R) Cuda compiler driver
...
release 12.4, V12.4.99
```

Check that `cudnn.h` exists:

```
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include\cudnn.h"
```

---

## Step 1 — Download the CIFAR-10 dataset

CIFAR-10 is distributed as a binary file archive. Download the **binary version** (not the Python pickle version):

**URL:** `https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`

### Option A — Use the data that already exists in this repo

If you already ran `CPP_ResNet`, the data is already extracted at:

```
CUDA_ResNet/data/cifar-10-batches-bin/
```

You can point `CUDA_ResNet` directly at this path (see Step 4). No extra download needed.

### Option B — Fresh download

1. Download `cifar-10-binary.tar.gz` from the URL above.

2. Extract it. On Windows, use 7-Zip or WSL:

```bash
# WSL / Git Bash
tar -xzf cifar-10-binary.tar.gz
```

3. Place the extracted folder so the layout looks exactly like this:

```
<your-data-root>/
└── cifar-10-batches-bin/
    ├── data_batch_1.bin     ← 30 MB each, 10,000 training images
    ├── data_batch_2.bin
    ├── data_batch_3.bin
    ├── data_batch_4.bin
    ├── data_batch_5.bin
    ├── test_batch.bin       ← 10,000 test images
    ├── batches.meta.txt
    └── readme.html
```

The loader reads exactly these filenames. The path you pass at runtime must be the **parent** of `cifar-10-batches-bin/`, i.e., `<your-data-root>`.

Each `.bin` file uses the CIFAR-10 binary format:
- 10,000 records, each 3073 bytes
- Byte 0: class label (0–9)
- Bytes 1–3072: RGB pixel values in CHW order, `uint8`

---

## Step 2 — Set your GPU architecture

Open `CMakeLists.txt` and find this line near the top:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)   # Change to match your GPU
```

Replace `86` with the compute capability of your GPU:

| GPU family | Compute Capability |
|---|---|
| RTX 40xx (Ada Lovelace) | `89` |
| RTX 30xx (Ampere) | `86` |
| RTX 20xx / GTX 16xx (Turing) | `75` |
| A100 | `80` |
| V100 | `70` |
| H100 | `90` |

To find your GPU's compute capability:

```
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

If your CUDA toolkit is not at the default path `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4`, also update this line in `CMakeLists.txt`:

```cmake
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
```

---

## Step 3 — Build

Open a **Developer Command Prompt for VS 2022** (or VS 2019). This sets up MSVC environment variables that CMake and NVCC need.

Navigate to the `CUDA_ResNet` folder:

```
cd "C:\Users\karan\Documents\Optimization Techniques\CUDA_ResNet"
```

Create a build directory and configure:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

You should see output like:

```
-- Build type  : Release
-- CUDA arch   : 86
-- CUDA toolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4
-- Configuring done
-- Build files have been written to: .../CUDA_ResNet/build
```

Compile (this takes 2–5 minutes; NVCC compiles each `.cu` file separately):

```
cmake --build . --config Release
```

The executable is placed at:

```
build\Release\CUDA_ResNet.exe
```

### Build troubleshooting

**`nvcc not found` or CMake can't find CUDA**

Make sure the CUDA bin directory is in your PATH. In Developer Command Prompt:

```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%
```

**`cudnn.h: No such file or directory`**

cuDNN headers ship inside the CUDA toolkit since CUDA 12.0. If you have an older separate cuDNN installation, add its include path to `CMakeLists.txt`:

```cmake
include_directories("C:/path/to/cudnn/include")
link_directories("C:/path/to/cudnn/lib/x64")
```

**`error: identifier "__float128" is undefined`**

This is a known MSVC + NVCC interaction. Add this to `CMakeLists.txt`:

```cmake
add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
```

**Missing DLLs at runtime**

The `CMakeLists.txt` copies `cudnn64_9.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll` next to the `.exe` automatically. If your cuDNN version differs, update the DLL names in `CMakeLists.txt`:

```cmake
foreach(dll cudnn64_9 cublas64_12 cublasLt64_12)
```

Replace `cudnn64_9` with the actual DLL name in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\`.

---

## Step 4 — Run

From inside the `build` directory:

```
.\Release\CUDA_ResNet.exe <path-to-data-root>
```

If you omit the path argument, it defaults to that same location:

```
.\Release\CUDA_ResNet.exe
```

### Expected startup output

```
GPU: NVIDIA GeForce RTX 3080  (CC 8.6)
Loading CIFAR-10 from C:/Users/karan/Documents/Optimization Techniques/CUDA_ResNet/data ...
Train: 50000  Test: 10000
Initialising ResNet-18 (benchmarking cuDNN algorithms)...
Model ready.
```

The `benchmarking cuDNN algorithms` step runs `cudnnFindConvolutionForwardAlgorithm` for every convolutional layer in the network (20 convolutions in total — 1 stem + 16 BasicBlock convs + 3 shortcut convs). This happens once and takes 5–30 seconds depending on your GPU. The selected algorithms are then used for every subsequent forward and backward pass.

### Expected training output

```
Epoch  1/30 | lr=0.10000 | Train Loss=1.8432 Acc=32.14% | Val Loss=1.6201 Acc=40.22% | Train 18.3s | Total 21.1s
Epoch  2/30 | lr=0.10000 | Train Loss=1.4871 Acc=45.89% | Val Loss=1.3944 Acc=49.76% | Train 17.8s | Total 38.9s
...
Epoch 30/30 | lr=0.00001 | Train Loss=0.1823 Acc=93.61% | Val Loss=0.5412 Acc=88.74% | Train 17.6s | Total 591s
```

A progress bar prints within each epoch:

```
  [782/782] loss=0.1521 Iter/sec=46.2
```

### Typical convergence (ResNet-18, CIFAR-10, 30 epochs)

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 5 | ~70% | ~68% |
| 10 | ~82% | ~78% |
| 20 | ~92% | ~87% |
| 30 | ~94% | ~89% |

---

## Hyperparameters

All hyperparameters are constants at the top of `src/main.cu`. Edit and recompile to change them:

```cpp
const int   BATCH    = 64;     // Batch size
const int   EPOCHS   = 30;     // Training epochs
const float BASE_LR  = 0.1f;  // Initial learning rate
const float MOMENTUM = 0.9f;  // SGD momentum (Nesterov)
const float WD       = 5e-4f; // Weight decay (L2 regularisation)
```

**Learning rate schedule** — StepLR, matching `CPP_ResNet`:

```cpp
// step_size=5, gamma=0.1
// lr = BASE_LR * 0.1^(epoch // 5)
```

| Epoch range | Learning rate |
|---|---|
| 1–4 | 0.1 |
| 5–9 | 0.01 |
| 10–14 | 0.001 |
| 15–19 | 0.0001 |
| 20+ | 0.00001 |

---

## Architecture

ResNet-18 adapted for CIFAR-10 (same as `CPP_ResNet`):

```
Input (N, 3, 32, 32)
  └─ Stem Conv 3→64, 3×3, stride=1, pad=1 → BN → ReLU
  └─ Stage 1: 2× BasicBlock(64→64,  stride=1)    → (N,  64, 32, 32)
  └─ Stage 2: 2× BasicBlock(64→128, stride=2)    → (N, 128, 16, 16)
  └─ Stage 3: 2× BasicBlock(128→256, stride=2)   → (N, 256,  8,  8)
  └─ Stage 4: 2× BasicBlock(256→512, stride=2)   → (N, 512,  4,  4)
  └─ AvgPool(4×4)                                 → (N, 512,  1,  1)
  └─ Linear(512 → 10)
```

Each BasicBlock:

```
x ──→ Conv(3×3) → BN → ReLU → Conv(3×3) → BN ──→ [Fused Add+ReLU] → y
│                                                          ↑
└─────────────────→ Shortcut (1×1 Conv + BN if dim changes) ┘
```

The **Fused Add+ReLU** is our custom CUDA kernel (`kernels.cu`). It replaces two separate cuDNN calls (elementwise add, then ReLU) with one kernel that reads each element once and writes once.

**Total parameters:** ~11.2 million (identical to `CPP_ResNet`)

---

## GPU memory usage

At batch size 64:

| Component | Approximate size |
|---|---|
| Model weights | ~44 MB |
| Activations (forward pass) | ~120 MB |
| Gradients | ~120 MB |
| cuDNN workspaces | ~50–200 MB |
| Dataset (pinned host) | ~750 MB (train) + ~150 MB (test) |
| **GPU total** | **~350–550 MB** |

A GPU with **6 GB VRAM** or more is sufficient. If you run out of memory, reduce `BATCH` from 64 to 32 in `main.cu`.

---

## Profiling with Nsight Systems

The binary is compiled with `-lineinfo`, which preserves source-level correlation in the profiler.

```
nsys profile --trace=cuda,cudnn,cublas --output=cuda_resnet_profile .\Release\CUDA_ResNet.exe
```

Open the resulting `.nsys-rep` file in Nsight Systems to see:
- cuDNN kernel execution on the GPU timeline
- PCIe transfer overlap with compute (the double-buffer streams)
- Custom kernel contributions (`k_add_relu`, `k_sgd`, `k_cross_entropy`)

---

## Project structure

```
CUDA_ResNet/
├── CMakeLists.txt          — Build system; set CMAKE_CUDA_ARCHITECTURES here
├── include/
│   ├── cuda_check.cuh      — CUDA_CHECK / CUDNN_CHECK / CUBLAS_CHECK macros
│   └── resnet.cuh          — All struct declarations
└── src/
    ├── kernels.cu          — Custom CUDA kernels (fused add+relu, SGD, loss, normalise)
    ├── layers.cu           — cuDNN conv/BN/pool + cuBLAS linear; algorithm benchmarking
    ├── resnet.cu           — BasicBlock and ResNet18 (forward, backward, update)
    ├── cifar10.cpp         — Binary loader with pinned host memory
    └── main.cu             — Training loop; CUDA streams; hyperparameters
```

---

## Comparison with CPP_ResNet

Both projects train the identical ResNet-18 architecture on CIFAR-10 with the same hyperparameters.

| Feature | CPP_ResNet (LibTorch) | CUDA_ResNet (raw CUDA) |
|---|---|---|
| Convolutions | cuDNN via LibTorch | cuDNN directly; benchmarked per layer |
| BatchNorm | cuDNN via LibTorch | cuDNN directly |
| Residual Add + ReLU | Two separate kernels | One fused custom kernel |
| Data loading | Pageable CPU memory | Pinned (`cudaMallocHost`) |
| Batch prefetch | No | Yes — CUDA stream overlap |
| Autograd | Yes (dynamic graph) | No — manual backward |
| Normalisation | CPU side (dataset transform) | GPU side (custom kernel) |
| Dependencies | LibTorch (~2 GB) | CUDA Toolkit only |
