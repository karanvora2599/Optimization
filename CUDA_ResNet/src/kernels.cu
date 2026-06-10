// kernels.cu — Custom CUDA kernels that bypass LibTorch:
//   • Fused Add+ReLU (saves a memory round-trip vs two separate kernels)
//   • Cross-entropy (log-softmax + NLL in one pass)
//   • Nesterov SGD update
//   • Data normalisation (GPU-side, avoids a CPU preprocessing step)

#include "../include/cuda_check.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

// ── Helpers ──────────────────────────────────────────────────────────────────

static constexpr int BLOCK = 256;

// ── ReLU ─────────────────────────────────────────────────────────────────────

__global__ void k_relu(const float *__restrict__ x,
                        float *__restrict__ y,
                        bool  *__restrict__ mask, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v  = x[i];
    mask[i]  = (v > 0.f);
    y[i]     = v > 0.f ? v : 0.f;
}

__global__ void k_relu_bwd(const float *__restrict__ dy,
                            const bool  *__restrict__ mask,
                            float       *__restrict__ dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] = mask[i] ? dy[i] : 0.f;
}

void launch_relu(const float *x, float *y, bool *mask, int n, cudaStream_t s)
{
    k_relu<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(x, y, mask, n);
}

void launch_relu_backward(const float *dy, const bool *mask, float *dx, int n, cudaStream_t s)
{
    k_relu_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(dy, mask, dx, n);
}

// ── Fused Add + ReLU ─────────────────────────────────────────────────────────
// Saves one global-memory round-trip vs cuDNN dispatching BN+add and ReLU as
// separate kernels.

__global__ void k_add_relu(const float *__restrict__ a,
                            const float *__restrict__ b,
                            float       *__restrict__ y,
                            bool        *__restrict__ mask, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i] + b[i];
    mask[i] = (v > 0.f);
    y[i]    = v > 0.f ? v : 0.f;
}

// Gradient of add + ReLU: both inputs get the same masked gradient.
__global__ void k_add_relu_bwd(const float *__restrict__ dy,
                                const bool  *__restrict__ mask,
                                float *__restrict__ da,
                                float *__restrict__ db, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = mask[i] ? dy[i] : 0.f;
    da[i] = g;
    db[i] = g;
}

void launch_add_relu(const float *a, const float *b, float *y, bool *mask,
                     int n, cudaStream_t s)
{
    k_add_relu<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(a, b, y, mask, n);
}

void launch_add_relu_backward(const float *dy, const bool *mask,
                               float *da, float *db, int n, cudaStream_t s)
{
    k_add_relu_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(dy, mask, da, db, n);
}

// ── Cross-entropy (numerically stable softmax + NLL) ─────────────────────────
// One kernel per sample — for batch≤256 and C=10 this fits nicely in shared mem.
// Each block handles one sample using warp-level reductions.

__global__ void k_cross_entropy(const float *__restrict__ logits,  // (N, C)
                                 const int   *__restrict__ labels,  // (N,)
                                 float       *__restrict__ probs,   // (N, C) — softmax
                                 float       *__restrict__ loss_buf,// (N,) — per-sample loss
                                 int N, int C)
{
    int n = blockIdx.x;
    if (n >= N) return;

    const float *row_logits = logits + n * C;
    float       *row_probs  = probs  + n * C;

    // Find max for numerical stability (use shared memory for small C)
    extern __shared__ float smem[];
    float mx = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        mx = fmaxf(mx, row_logits[c]);
    smem[threadIdx.x] = mx;
    __syncthreads();
    // Reduce max across block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    // Compute exp(x - max) and accumulate sum
    float sumexp = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float e = expf(row_logits[c] - mx);
        row_probs[c] = e;
        smem[threadIdx.x] += e;
    }
    smem[threadIdx.x] = sumexp;
    // reset for sum reduction
    if (threadIdx.x == 0) smem[0] = 0.f;
    __syncthreads();
    atomicAdd(&smem[0], sumexp);
    __syncthreads();

    // Not ideal with atomics — redo properly:
    // Reset smem
    smem[threadIdx.x] = 0.f;
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        smem[threadIdx.x] += row_probs[c];
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float inv_sum = 1.f / smem[0];

    // Normalise probs
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        row_probs[c] *= inv_sum;
    __syncthreads();

    // NLL loss for the true class
    if (threadIdx.x == 0) {
        int label = labels[n];
        loss_buf[n] = -logf(fmaxf(row_probs[label], 1e-7f));
    }
}

// Backward: d_logits[n,c] = (probs[n,c] - (c==label)) / N
__global__ void k_cross_entropy_bwd(const float *__restrict__ probs,
                                     const int   *__restrict__ labels,
                                     float       *__restrict__ d_logits,
                                     int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N*C) return;
    int n = idx / C, c = idx % C;
    float p = probs[n*C + c];
    d_logits[idx] = (p - (float)(c == labels[n])) / (float)N;
}

void launch_cross_entropy(const float *logits, const int *labels,
                           float *probs, float *loss_buf,
                           int N, int C, cudaStream_t s)
{
    // One block per sample, 32 threads per block, shared mem = 32 floats
    k_cross_entropy<<<N, 32, 32*sizeof(float), s>>>(logits, labels, probs, loss_buf, N, C);
}

void launch_cross_entropy_backward(const float *probs, const int *labels,
                                   float *d_logits, int N, int C, cudaStream_t s)
{
    int total = N * C;
    k_cross_entropy_bwd<<<(total+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(probs, labels, d_logits, N, C);
}

// ── Nesterov SGD ─────────────────────────────────────────────────────────────
// Matches PyTorch SGD(momentum, nesterov=True, weight_decay):
//   g   = dw + wd * w
//   v   = mom * v + g
//   g_n = g + mom * v         (Nesterov lookahead)
//   w  -= lr * g_n

__global__ void k_sgd(float *__restrict__ w,
                       float *__restrict__ dw,
                       float *__restrict__ v,
                       float lr, float mom, float wd, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = dw[i] + wd * w[i];
    float vi = mom * v[i] + g;
    v[i]     = vi;
    float gn = g + mom * vi;
    w[i]    -= lr * gn;
    dw[i]    = 0.f;   // zero gradient in-place
}

void launch_sgd(float *w, float *dw, float *v,
                float lr, float mom, float wd, int n, cudaStream_t s)
{
    k_sgd<<<(n+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(w, dw, v, lr, mom, wd, n);
}

// ── GPU-side normalisation (uint8→float, CIFAR-10) ────────────────────────────
// Avoids a redundant CPU pass — data arrives as float [0,1] from the loader,
// we apply per-channel mean/std on the GPU.

__global__ void k_normalize(float *__restrict__ data,
                             const float *__restrict__ mean,
                             const float *__restrict__ std_dev,
                             int N, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx >= total) return;
    int c    = (idx / HW) % C;
    data[idx] = (data[idx] - mean[c]) / std_dev[c];
}

void launch_normalize(float *data, const float *mean, const float *std_dev,
                      int N, int C, int HW, cudaStream_t s)
{
    int total = N * C * HW;
    k_normalize<<<(total+BLOCK-1)/BLOCK, BLOCK, 0, s>>>(data, mean, std_dev, N, C, HW);
}
