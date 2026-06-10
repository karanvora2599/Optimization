// layers.cu — Direct cuDNN + cuBLAS layer implementations.
// Bypasses LibTorch entirely: we call cuDNN/cuBLAS APIs ourselves, giving us:
//   • Choice of algorithm (benchmarked at init time)
//   • No autograd overhead
//   • Easier fusion points for custom kernels

#include "../include/cuda_check.cuh"
#include "../include/resnet.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>

cudnnHandle_t  g_cudnn  = nullptr;
cublasHandle_t g_cublas = nullptr;

static const float ONE  = 1.f;
static const float ZERO = 0.f;

// ─────────────────────────────────────────────────────────────────────────────
// ConvLayer
// ─────────────────────────────────────────────────────────────────────────────

void ConvLayer::init(int ic, int oc, int kh, int kw, int p, int st,
                     int N, int H, int W)
{
    in_c = ic; out_c = oc; kH = kh; kW = kw; pad = p; stride = st;

    // Allocate weights + grads + velocity
    size_t wsz = (size_t)oc * ic * kh * kw;
    CUDA_CHECK(cudaMalloc(&w,  wsz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, wsz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vw, wsz * sizeof(float)));
    CUDA_CHECK(cudaMemset(vw, 0, wsz * sizeof(float)));

    // Xavier uniform initialisation on CPU, then copy
    float *h = new float[wsz];
    float limit = sqrtf(6.f / (float)(ic * kh * kw + oc));
    for (size_t i = 0; i < wsz; i++)
        h[i] = ((float)rand() / RAND_MAX * 2.f - 1.f) * limit;
    CUDA_CHECK(cudaMemcpy(w, h, wsz * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h;
    CUDA_CHECK(cudaMemset(dw, 0, wsz * sizeof(float)));

    // cuDNN descriptors
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&flt_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    int outH = (H + 2*p - kh) / st + 1;
    int outW = (W + 2*p - kw) / st + 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, ic, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, oc, outH, outW));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(flt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, oc, ic, kh, kw));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, p, p, st, st, 1, 1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH)); // allow Tensor Cores

    // Benchmark to find fastest algorithm
    int algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf[8];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        g_cudnn, in_desc, flt_desc, conv_desc, out_desc,
        8, &algo_count, fwd_perf));
    fwd_algo = fwd_perf[0].algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        g_cudnn, in_desc, flt_desc, conv_desc, out_desc, fwd_algo, &fwd_ws_sz));
    if (fwd_ws_sz > 0) CUDA_CHECK(cudaMalloc(&fwd_ws, fwd_ws_sz));

    cudnnConvolutionBwdDataAlgoPerf_t bd_perf[8];
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
        g_cudnn, flt_desc, out_desc, conv_desc, in_desc,
        8, &algo_count, bd_perf));
    bwd_d_algo = bd_perf[0].algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        g_cudnn, flt_desc, out_desc, conv_desc, in_desc, bwd_d_algo, &bwd_d_ws_sz));
    if (bwd_d_ws_sz > 0) CUDA_CHECK(cudaMalloc(&bwd_d_ws, bwd_d_ws_sz));

    cudnnConvolutionBwdFilterAlgoPerf_t bf_perf[8];
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
        g_cudnn, in_desc, out_desc, conv_desc, flt_desc,
        8, &algo_count, bf_perf));
    bwd_f_algo = bf_perf[0].algo;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        g_cudnn, in_desc, out_desc, conv_desc, flt_desc, bwd_f_algo, &bwd_f_ws_sz));
    if (bwd_f_ws_sz > 0) CUDA_CHECK(cudaMalloc(&bwd_f_ws, bwd_f_ws_sz));
}

void ConvLayer::forward(const float *x, float *y)
{
    CUDNN_CHECK(cudnnConvolutionForward(
        g_cudnn, &ONE, in_desc, x, flt_desc, w, conv_desc,
        fwd_algo, fwd_ws, fwd_ws_sz,
        &ZERO, out_desc, y));
}

void ConvLayer::backward(const float *dy, float *dx, const float *x)
{
    // Gradient w.r.t. filter weights (accumulate into dw)
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
        g_cudnn, &ONE, in_desc, x, out_desc, dy, conv_desc,
        bwd_f_algo, bwd_f_ws, bwd_f_ws_sz,
        &ONE, flt_desc, dw));   // beta=1 → accumulate

    // Gradient w.r.t. input (dx may be nullptr at first layer)
    if (dx) {
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            g_cudnn, &ONE, flt_desc, w, out_desc, dy, conv_desc,
            bwd_d_algo, bwd_d_ws, bwd_d_ws_sz,
            &ZERO, in_desc, dx));
    }
}

void ConvLayer::zero_grad()
{
    CUDA_CHECK(cudaMemset(dw, 0, (size_t)out_c * in_c * kH * kW * sizeof(float)));
}

void ConvLayer::free_mem()
{
    cudaFree(w);  cudaFree(dw);  cudaFree(vw);
    if (fwd_ws)    cudaFree(fwd_ws);
    if (bwd_d_ws)  cudaFree(bwd_d_ws);
    if (bwd_f_ws)  cudaFree(bwd_f_ws);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(flt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// ─────────────────────────────────────────────────────────────────────────────
// BNLayer
// ─────────────────────────────────────────────────────────────────────────────

void BNLayer::init(int c, int N, int H, int W)
{
    C = c;
    CUDA_CHECK(cudaMalloc(&scale,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dscale, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vscale, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bias,   C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dbias,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vbias,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rmean,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rvar,   C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&saved_mean, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&saved_ivar, C * sizeof(float)));

    // Init scale=1, bias=0, running stats=0/1
    float *ones  = new float[C];
    float *zeros = new float[C];
    for (int i = 0; i < C; i++) { ones[i] = 1.f; zeros[i] = 0.f; }
    CUDA_CHECK(cudaMemcpy(scale, ones,  C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rvar,  ones,  C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(bias,  0, C * sizeof(float)));
    CUDA_CHECK(cudaMemset(rmean, 0, C * sizeof(float)));
    CUDA_CHECK(cudaMemset(vscale, 0, C * sizeof(float)));
    CUDA_CHECK(cudaMemset(vbias,  0, C * sizeof(float)));
    delete[] ones;
    delete[] zeros;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, CUDNN_BATCHNORM_SPATIAL));
}

void BNLayer::forward_train(const float *x, float *y)
{
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        g_cudnn, CUDNN_BATCHNORM_SPATIAL,
        &ONE, &ZERO,
        x_desc, x,
        x_desc, y,
        bn_desc, scale, bias,
        momentum,
        rmean, rvar,
        eps,
        saved_mean, saved_ivar));
}

void BNLayer::forward_infer(const float *x, float *y)
{
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        g_cudnn, CUDNN_BATCHNORM_SPATIAL,
        &ONE, &ZERO,
        x_desc, x,
        x_desc, y,
        bn_desc, scale, bias,
        rmean, rvar, eps));
}

void BNLayer::backward(const float *dy, const float *x, float *dx)
{
    CUDNN_CHECK(cudnnBatchNormalizationBackward(
        g_cudnn, CUDNN_BATCHNORM_SPATIAL,
        &ONE, &ZERO,   // alpha/beta for dx
        &ONE, &ONE,    // alpha/beta for dscale/dbias (accumulate)
        x_desc, x,
        x_desc, dy,
        x_desc, dx,
        bn_desc, scale,
        dscale, dbias,
        eps,
        saved_mean, saved_ivar));
}

void BNLayer::zero_grad()
{
    CUDA_CHECK(cudaMemset(dscale, 0, C * sizeof(float)));
    CUDA_CHECK(cudaMemset(dbias,  0, C * sizeof(float)));
}

void BNLayer::free_mem()
{
    cudaFree(scale);  cudaFree(dscale);  cudaFree(vscale);
    cudaFree(bias);   cudaFree(dbias);   cudaFree(vbias);
    cudaFree(rmean);  cudaFree(rvar);
    cudaFree(saved_mean);  cudaFree(saved_ivar);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
}

// ─────────────────────────────────────────────────────────────────────────────
// PoolAvgLayer
// ─────────────────────────────────────────────────────────────────────────────

void PoolAvgLayer::init(int kh, int kw, int st,
                        int N, int C, int H, int W)
{
    int outH = (H - kh) / st + 1;
    int outW = (W - kw) / st + 1;

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(pool_desc,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        kh, kw, 0, 0, st, st));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, outH, outW));
}

void PoolAvgLayer::forward(const float *x, float *y)
{
    CUDNN_CHECK(cudnnPoolingForward(g_cudnn, pool_desc,
        &ONE, in_desc, x, &ZERO, out_desc, y));
}

void PoolAvgLayer::backward(const float *dy, const float *x, const float *y, float *dx)
{
    CUDNN_CHECK(cudnnPoolingBackward(g_cudnn, pool_desc,
        &ONE,  out_desc, y,
               out_desc, dy,
               in_desc,  x,
        &ZERO, in_desc,  dx));
}

void PoolAvgLayer::free_mem()
{
    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearLayer  (uses cuBLAS SGEMM)
// ─────────────────────────────────────────────────────────────────────────────

void LinearLayer::init(int inf, int outf)
{
    in_f = inf; out_f = outf;
    CUDA_CHECK(cudaMalloc(&w,  (size_t)outf * inf  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, (size_t)outf * inf  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vw, (size_t)outf * inf  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b,  (size_t)outf        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, (size_t)outf        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vb, (size_t)outf        * sizeof(float)));

    // Kaiming uniform for weights
    float *hw = new float[(size_t)outf * inf];
    float limit = sqrtf(1.f / inf);
    for (int i = 0; i < outf * inf; i++)
        hw[i] = ((float)rand() / RAND_MAX * 2.f - 1.f) * limit;
    CUDA_CHECK(cudaMemcpy(w, hw, (size_t)outf * inf * sizeof(float), cudaMemcpyHostToDevice));
    delete[] hw;

    CUDA_CHECK(cudaMemset(b,  0, outf * sizeof(float)));
    CUDA_CHECK(cudaMemset(vw, 0, (size_t)outf * inf * sizeof(float)));
    CUDA_CHECK(cudaMemset(vb, 0, outf * sizeof(float)));
    CUDA_CHECK(cudaMemset(dw, 0, (size_t)outf * inf * sizeof(float)));
    CUDA_CHECK(cudaMemset(db, 0, outf * sizeof(float)));
}

// y = x * W^T + b
// x: (N, in_f),  W: (out_f, in_f),  y: (N, out_f)
// Using cuBLAS column-major convention: y^T = W * x^T
void LinearLayer::forward(const float *x, float *y, int N)
{
    // y = x * W^T  →  cuBLAS: C = A*B  with A=W(out_f×in_f), B=x^T(in_f×N) → col-major
    // We compute y^T(out_f × N) = W(out_f × in_f) * x^T(in_f × N)
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_T,  // W already stored as (out_f, in_f) row-major = (in_f, out_f) col-major → need no transpose
        CUBLAS_OP_N,
        out_f, N, in_f,
        &ONE,
        w, in_f,           // W col-major: in_f rows, out_f cols
        x, in_f,           // x col-major: in_f rows, N cols
        &ZERO,
        y, out_f));        // y col-major: out_f rows, N cols

    // Add bias: y[n,c] += b[c]
    // cuBLAS doesn't do broadcast directly; use a ones vector trick
    // OR use a simple custom kernel (for out_f=10 this is trivial)
    // Simple approach: add bias column-wise via cudnnAddTensor
    // For simplicity, use a loop in a __global__ (out_f=10, negligible)
    // We'll do it with CUDA kernel defined inline:
    // Launch N*out_f threads
    struct AddBias {
        static __global__ void run(float *y, const float *b, int N, int outf) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N * outf) return;
            // y is (out_f, N) col-major from cuBLAS — need to transpose access
            int n = i / outf, c = i % outf;
            y[c * N + n] += b[c];
        }
    };
    int tot = N * out_f;
    AddBias::run<<<(tot+255)/256, 256>>>(y, b, N, out_f);
}

// dx = dy * W  (N×out_f → N×in_f),  dW += dy^T * x,  db += sum(dy, dim=0)
void LinearLayer::backward(const float *dy, const float *x, float *dx, int N)
{
    // y is (out_f × N) col-major, dy same shape

    // dx = W^T * dy  →  (in_f × N) = (in_f × out_f) * (out_f × N)
    if (dx) {
        CUBLAS_CHECK(cublasSgemm(g_cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            in_f, N, out_f,
            &ONE,
            w, in_f,
            dy, out_f,
            &ZERO,
            dx, in_f));
    }

    // dW += dy * x^T / N  →  (out_f × in_f) = (out_f × N) * (N × in_f)
    CUBLAS_CHECK(cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_T,
        out_f, in_f, N,
        &ONE,
        dy, out_f,
        x, in_f,
        &ONE,        // accumulate
        dw, out_f));

    // db += sum of each row of dy  (dy is out_f × N col-major)
    // Use cuBLAS gemv with a ones vector
    struct BiasGrad {
        static __global__ void run(const float *dy, float *db, int N, int outf) {
            int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= outf) return;
            float s = 0.f;
            for (int n = 0; n < N; n++) s += dy[c * N + n]; // col-major access
            db[c] += s;
        }
    };
    BiasGrad::run<<<(out_f+255)/256, 256>>>(dy, db, N, out_f);
}

void LinearLayer::zero_grad()
{
    CUDA_CHECK(cudaMemset(dw, 0, (size_t)out_f * in_f * sizeof(float)));
    CUDA_CHECK(cudaMemset(db, 0, out_f * sizeof(float)));
}

void LinearLayer::free_mem()
{
    cudaFree(w);  cudaFree(dw);  cudaFree(vw);
    cudaFree(b);  cudaFree(db);  cudaFree(vb);
}
