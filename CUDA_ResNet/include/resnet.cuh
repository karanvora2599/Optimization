#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdint>

// ── Global handles (initialized in main.cu) ──────────────────────────────────
extern cudnnHandle_t  g_cudnn;
extern cublasHandle_t g_cublas;

// ── Convolution layer (no bias, as in ResNet) ────────────────────────────────
struct ConvLayer {
    float *w{nullptr}, *dw{nullptr}, *vw{nullptr};   // weight, grad, velocity
    int in_c, out_c, kH, kW, pad, stride;

    cudnnTensorDescriptor_t  in_desc{nullptr}, out_desc{nullptr};
    cudnnFilterDescriptor_t  flt_desc{nullptr};
    cudnnConvolutionDescriptor_t conv_desc{nullptr};

    cudnnConvolutionFwdAlgo_t      fwd_algo{};
    cudnnConvolutionBwdDataAlgo_t  bwd_d_algo{};
    cudnnConvolutionBwdFilterAlgo_t bwd_f_algo{};

    void *fwd_ws{nullptr},  *bwd_d_ws{nullptr}, *bwd_f_ws{nullptr};
    size_t fwd_ws_sz{0},    bwd_d_ws_sz{0},     bwd_f_ws_sz{0};

    void init(int in_c, int out_c, int kH, int kW, int pad, int stride,
              int N, int H, int W);
    void forward (const float *x, float *y);
    // dx may be nullptr (first layer); x is needed for bwd_filter
    void backward(const float *dy, float *dx, const float *x);
    void zero_grad();
    void free_mem();
};

// ── Batch Normalisation layer ─────────────────────────────────────────────────
struct BNLayer {
    float *scale{nullptr},  *dscale{nullptr},  *vscale{nullptr};
    float *bias{nullptr},   *dbias{nullptr},   *vbias{nullptr};
    float *rmean{nullptr},  *rvar{nullptr};        // running statistics
    float *saved_mean{nullptr}, *saved_ivar{nullptr}; // saved for backward
    int C;
    double eps{1e-5}, momentum{0.1};

    cudnnTensorDescriptor_t x_desc{nullptr}, bn_desc{nullptr};

    void init(int C, int N, int H, int W);
    void forward_train(const float *x, float *y);
    void forward_infer(const float *x, float *y);
    void backward(const float *dy, const float *x, float *dx);
    void zero_grad();
    void free_mem();
};

// ── Average pooling layer ─────────────────────────────────────────────────────
struct PoolAvgLayer {
    cudnnPoolingDescriptor_t pool_desc{nullptr};
    cudnnTensorDescriptor_t  in_desc{nullptr}, out_desc{nullptr};

    void init(int kH, int kW, int stride, int N, int C, int H, int W);
    void forward (const float *x, float *y);
    void backward(const float *dy, const float *x, const float *y, float *dx);
    void free_mem();
};

// ── Fully-connected layer ─────────────────────────────────────────────────────
struct LinearLayer {
    float *w{nullptr},  *dw{nullptr},  *vw{nullptr};  // (out_f × in_f)
    float *b{nullptr},  *db{nullptr},  *vb{nullptr};  // (out_f,)
    int in_f, out_f;

    void init(int in_f, int out_f);
    // x: (N, in_f)  y: (N, out_f)
    void forward (const float *x, float *y, int N);
    void backward(const float *dy, const float *x, float *dx, int N);
    void zero_grad();
    void free_mem();
};

// ── BasicBlock (two 3×3 convs + optional shortcut) ───────────────────────────
struct BasicBlock {
    ConvLayer conv1, conv2;
    BNLayer   bn1,   bn2;
    bool      has_sc{false};
    ConvLayer sc_conv;
    BNLayer   sc_bn;

    // Intermediate activations (allocated in init)
    float *buf_after_conv1{nullptr};   // (N, out_c, H, W)  — after conv1, before BN1
    float *buf_after_relu1{nullptr};   // (N, out_c, H, W)  — after BN1+ReLU
    float *buf_sc_out{nullptr};        // (N, out_c, outH, outW) — shortcut output
    bool  *relu1_mask{nullptr};        // elementwise mask for ReLU1 backward
    bool  *relu2_mask{nullptr};        // elementwise mask for final ReLU backward
    float *buf_bn2_out{nullptr};       // after bn2, before add+relu

    int N, in_c, out_c, inH, inW, outH, outW;

    void init(int in_c, int out_c, int stride, int N, int inH, int inW);
    void forward      (const float *x, float *y);
    void forward_infer(const float *x, float *y);
    void backward(const float *dy, float *dx, const float *x);
    void zero_grad();
    void free_mem();
};

// ── ResNet-18 ─────────────────────────────────────────────────────────────────
struct ResNet18 {
    // Stem
    ConvLayer stem_conv;
    BNLayer   stem_bn;
    bool      *stem_relu_mask{nullptr};

    // 4 stages, 2 blocks each
    BasicBlock stage1[2], stage2[2], stage3[2], stage4[2];

    PoolAvgLayer avgpool;
    LinearLayer  fc;

    // Activation buffers — one per block boundary so backward has the correct x
    float *act_stem{nullptr};           // (N,  64, 32, 32)  — after stem
    float *act_s1_0{nullptr};           // (N,  64, 32, 32)  — output of stage1[0]
    float *act_s1{nullptr};             // (N,  64, 32, 32)  — output of stage1[1]
    float *act_s2_0{nullptr};           // (N, 128, 16, 16)  — output of stage2[0]
    float *act_s2{nullptr};             // (N, 128, 16, 16)  — output of stage2[1]
    float *act_s3_0{nullptr};           // (N, 256,  8,  8)  — output of stage3[0]
    float *act_s3{nullptr};             // (N, 256,  8,  8)  — output of stage3[1]
    float *act_s4_0{nullptr};           // (N, 512,  4,  4)  — output of stage4[0]
    float *act_s4{nullptr};             // (N, 512,  4,  4)  — output of stage4[1]
    float *act_pool{nullptr};           // (N, 512,  1,  1)
    float *logits{nullptr};             // (N,  10)

    // Gradient buffers (matching the activation buffers above)
    float *d_stem{nullptr};
    float *d_s1_0{nullptr}, *d_s1{nullptr};
    float *d_s2_0{nullptr}, *d_s2{nullptr};
    float *d_s3_0{nullptr}, *d_s3{nullptr};
    float *d_s4_0{nullptr}, *d_s4{nullptr};
    float *d_pool{nullptr};

    // Loss workspace
    float *probs{nullptr};             // (N, 10)  — softmax output
    float *h_loss{nullptr};            // pinned host scalar

    int N;   // batch size

    void init(int batch);
    void forward(const float *x_dev, bool training);
    float loss_and_grad(const int *labels_dev);
    void backward(const int *labels_dev);
    void backward_stem(const int *labels_dev, const float *x_dev);
    void update(float lr, float momentum, float wd);
    void zero_grad();
    void free_mem();
};

// ── Custom CUDA kernels (defined in kernels.cu) ───────────────────────────────
// Elementwise ReLU with mask storage
void launch_relu(const float *x, float *y, bool *mask, int n, cudaStream_t s=0);
void launch_relu_backward(const float *dy, const bool *mask, float *dx, int n, cudaStream_t s=0);

// Fused add + ReLU  (y = relu(a + b))
void launch_add_relu(const float *a, const float *b, float *y, bool *mask, int n, cudaStream_t s=0);
// Gradient of fused add + ReLU: da = db = dy * mask
void launch_add_relu_backward(const float *dy, const bool *mask,
                               float *da, float *db, int n, cudaStream_t s=0);

// Cross-entropy forward: fills probs (softmax), writes per-sample loss to loss_buf
void launch_cross_entropy(const float *logits, const int *labels,
                           float *probs, float *loss_buf,
                           int N, int C, cudaStream_t s=0);
// Cross-entropy backward: d_logits = (probs - one_hot) / N
void launch_cross_entropy_backward(const float *probs, const int *labels,
                                   float *d_logits, int N, int C, cudaStream_t s=0);

// Nesterov SGD: updates w and v in-place
void launch_sgd(float *w, float *dw, float *v,
                float lr, float mom, float wd, int n, cudaStream_t s=0);

// Normalize uint8 CIFAR image in-place to float32 on device
void launch_normalize(float *data, const float *mean, const float *std_dev,
                      int N, int C, int HW, cudaStream_t s=0);

// ── CIFAR-10 loader ───────────────────────────────────────────────────────────
struct CIFAR10Loader {
    float  *h_images{nullptr};   // pinned host float (N, 3, 32, 32)
    int    *h_labels{nullptr};   // pinned host int   (N,)
    size_t  num_samples{0};

    float  *d_images{nullptr};   // device float
    int    *d_labels{nullptr};   // device int

    void load(const char *root, bool train);
    void to_device(cudaStream_t s=0);
    // Fill d_images/d_labels from a random batch (shuffled)
    void random_batch(int *shuffle_idx, int start, int batch_size, cudaStream_t s=0);
    void free_mem();
};
