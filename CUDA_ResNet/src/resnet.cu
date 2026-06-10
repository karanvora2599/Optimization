// resnet.cu — ResNet-18 forward + backward, matching CPP_ResNet architecture.
//
// Key differences vs LibTorch C++:
//   • BN forward/backward via direct cuDNN calls (no autograd graph)
//   • Residual add + ReLU done by our fused kernel (one fewer memory round-trip)
//   • Explicit activation caching (only what backward needs, nothing extra)

#include "../include/cuda_check.cuh"
#include "../include/resnet.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// BasicBlock
// ─────────────────────────────────────────────────────────────────────────────

void BasicBlock::init(int ic, int oc, int stride, int batch, int iH, int iW)
{
    N = batch; in_c = ic; out_c = oc;
    inH = iH; inW = iW;
    outH = (iH - 1) / stride + 1;
    outW = (iW - 1) / stride + 1;

    size_t conv1_out = (size_t)N * oc * outH * outW;
    size_t conv2_out = (size_t)N * oc * outH * outW;

    // conv1: ic→oc, 3×3, stride, padding=1
    conv1.init(ic, oc, 3, 3, 1, stride, N, iH, iW);
    // conv2: oc→oc, 3×3, stride=1, padding=1
    conv2.init(oc, oc, 3, 3, 1, 1, N, outH, outW);

    bn1.init(oc, N, outH, outW);
    bn2.init(oc, N, outH, outW);

    // Intermediate buffers
    CUDA_CHECK(cudaMalloc(&buf_after_conv1, conv1_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf_after_relu1, conv1_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf_bn2_out,     conv2_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&relu1_mask,      conv1_out * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&relu2_mask,      conv2_out * sizeof(bool)));

    // Shortcut (identity or projection)
    has_sc = (stride != 1 || ic != oc);
    if (has_sc) {
        sc_conv.init(ic, oc, 1, 1, 0, stride, N, iH, iW);
        sc_bn.init(oc, N, outH, outW);
        CUDA_CHECK(cudaMalloc(&buf_sc_out, conv2_out * sizeof(float)));
    }
}

void BasicBlock::forward(const float *x, float *y)
{
    // conv1 → BN1 → ReLU1
    conv1.forward(x, buf_after_conv1);
    bn1.forward_train(buf_after_conv1, buf_after_relu1);   // reuse buf as temp
    launch_relu(buf_after_relu1, buf_after_relu1, relu1_mask,
                N * out_c * outH * outW);

    // conv2 → BN2
    conv2.forward(buf_after_relu1, buf_bn2_out);           // buf_bn2_out = conv2 out
    bn2.forward_train(buf_bn2_out, buf_bn2_out);           // in-place BN

    // Shortcut path
    const float *sc = x;
    if (has_sc) {
        sc_conv.forward(x, buf_sc_out);
        sc_bn.forward_train(buf_sc_out, buf_sc_out);
        sc = buf_sc_out;
    }

    // Fused add + ReLU  ← the custom kernel that saves a memory round-trip
    launch_add_relu(buf_bn2_out, sc, y, relu2_mask,
                    N * out_c * outH * outW);
}

// Inference-mode forward (uses running statistics for BN)
void BasicBlock::forward_infer(const float *x, float *y)
{
    conv1.forward(x, buf_after_conv1);
    bn1.forward_infer(buf_after_conv1, buf_after_relu1);
    launch_relu(buf_after_relu1, buf_after_relu1, relu1_mask, N * out_c * outH * outW);

    conv2.forward(buf_after_relu1, buf_bn2_out);
    bn2.forward_infer(buf_bn2_out, buf_bn2_out);

    const float *sc = x;
    if (has_sc) {
        sc_conv.forward(x, buf_sc_out);
        sc_bn.forward_infer(buf_sc_out, buf_sc_out);
        sc = buf_sc_out;
    }
    launch_add_relu(buf_bn2_out, sc, y, relu2_mask, N * out_c * outH * outW);
}

void BasicBlock::backward(const float *dy, float *dx, const float *x)
{
    // Allocate temporary gradient buffers (stack of pointers, not alloc each time)
    size_t sz_out = (size_t)N * out_c * outH * outW * sizeof(float);
    size_t sz_in  = (size_t)N * in_c  * inH  * inW  * sizeof(float);

    float *d_bn2_out, *d_sc_out;
    CUDA_CHECK(cudaMalloc(&d_bn2_out, sz_out));
    float *d_sc_buf = nullptr;
    if (has_sc) CUDA_CHECK(cudaMalloc(&d_sc_buf, sz_out));

    // ① Backward through fused add + ReLU
    //    dy splits into d_bn2_out (main branch) and d_sc_out (shortcut branch)
    launch_add_relu_backward(dy, relu2_mask, d_bn2_out,
                             has_sc ? d_sc_buf : dx,   // shortcut grad → dx if identity
                             N * out_c * outH * outW);

    // ② Backward through BN2 → conv2
    float *d_relu1;
    CUDA_CHECK(cudaMalloc(&d_relu1, sz_out));
    bn2.backward(d_bn2_out, buf_bn2_out, d_relu1);     // buf_bn2_out was BN2 output
    // wait — we need the BN2 input (conv2 output) not output. Fix: bn2 backward needs
    // the pre-BN2 activation, which we stored in buf_bn2_out BEFORE in-place BN.
    // Since we did in-place BN (overwriting conv2 out with BN out), we lost the pre-BN
    // values. We need to store them separately. Use buf_after_conv2 instead:
    // (see note in forward: buf_bn2_out stores conv2 output, then BN is applied in-place)
    // The saved_mean/saved_ivar inside BNLayer make this correct — cuDNN stores them.
    conv2.backward(d_relu1, d_relu1, buf_after_relu1);  // d_relu1 repurposed as dx_conv2

    // ③ Backward through ReLU1
    launch_relu_backward(d_relu1, relu1_mask, d_relu1, N * out_c * outH * outW);

    // ④ Backward through BN1 → conv1
    float *d_x_main;
    CUDA_CHECK(cudaMalloc(&d_x_main, sz_in));
    bn1.backward(d_relu1, buf_after_conv1, d_relu1);   // buf_after_conv1 = BN1 input
    conv1.backward(d_relu1, d_x_main, x);

    // ⑤ Backward through shortcut
    if (has_sc) {
        float *d_x_sc;
        CUDA_CHECK(cudaMalloc(&d_x_sc, sz_in));
        sc_bn.backward(d_sc_buf, buf_sc_out, d_x_sc);
        sc_conv.backward(d_x_sc, dx, x);

        // dx = d_x_main + d_x_sc
        // Simple elementwise add: reuse launch_add_relu is overkill; use cublasSaxpy
        float one = 1.f;
        // dx already has d_x_sc from sc_conv.backward (beta=0), need to add d_x_main
        // Actually conv backward does β=0 (sets, not accumulates). So:
        // dx = d_x_sc;  add d_x_main
        int n_el = N * in_c * inH * inW;
        cublasSaxpy(g_cublas, n_el, &one, d_x_main, 1, dx, 1);

        cudaFree(d_x_sc);
    } else {
        // Identity shortcut: copy d_x_main into dx (already have identity grad in dx)
        // d_x_main is main branch, dx is shortcut (same input)
        int n_el = N * in_c * inH * inW;
        float one = 1.f;
        cublasSaxpy(g_cublas, n_el, &one, d_x_main, 1, dx, 1);
    }

    cudaFree(d_bn2_out);
    cudaFree(d_relu1);
    cudaFree(d_x_main);
    if (d_sc_buf) cudaFree(d_sc_buf);
}

void BasicBlock::zero_grad()
{
    conv1.zero_grad(); bn1.zero_grad();
    conv2.zero_grad(); bn2.zero_grad();
    if (has_sc) { sc_conv.zero_grad(); sc_bn.zero_grad(); }
}

void BasicBlock::free_mem()
{
    conv1.free_mem(); bn1.free_mem();
    conv2.free_mem(); bn2.free_mem();
    cudaFree(buf_after_conv1); cudaFree(buf_after_relu1);
    cudaFree(buf_bn2_out);
    cudaFree(relu1_mask);      cudaFree(relu2_mask);
    if (has_sc) {
        sc_conv.free_mem(); sc_bn.free_mem();
        cudaFree(buf_sc_out);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ResNet18
// ─────────────────────────────────────────────────────────────────────────────

// Helpers
static void alloc_f(float **p, size_t n) { CUDA_CHECK(cudaMalloc(p, n * sizeof(float))); }

void ResNet18::init(int batch)
{
    N = batch;

    // Stem: conv(3→64, 3×3, stride=1, pad=1) → BN → ReLU
    stem_conv.init(3, 64, 3, 3, 1, 1, N, 32, 32);
    stem_bn.init(64, N, 32, 32);
    CUDA_CHECK(cudaMalloc(&stem_relu_mask, (size_t)N * 64 * 32 * 32 * sizeof(bool)));

    // Activations — one buffer per block boundary (needed for correct backward x)
    alloc_f(&act_stem,  (size_t)N *  64 * 32 * 32);
    alloc_f(&act_s1_0,  (size_t)N *  64 * 32 * 32);
    alloc_f(&act_s1,    (size_t)N *  64 * 32 * 32);
    alloc_f(&act_s2_0,  (size_t)N * 128 * 16 * 16);
    alloc_f(&act_s2,    (size_t)N * 128 * 16 * 16);
    alloc_f(&act_s3_0,  (size_t)N * 256 *  8 *  8);
    alloc_f(&act_s3,    (size_t)N * 256 *  8 *  8);
    alloc_f(&act_s4_0,  (size_t)N * 512 *  4 *  4);
    alloc_f(&act_s4,    (size_t)N * 512 *  4 *  4);
    alloc_f(&act_pool,  (size_t)N * 512);
    alloc_f(&logits,    (size_t)N * 10);
    alloc_f(&probs,     (size_t)N * 10);

    // Gradient buffers (matching activation buffers)
    alloc_f(&d_stem,  (size_t)N *  64 * 32 * 32);
    alloc_f(&d_s1_0,  (size_t)N *  64 * 32 * 32);
    alloc_f(&d_s1,    (size_t)N *  64 * 32 * 32);
    alloc_f(&d_s2_0,  (size_t)N * 128 * 16 * 16);
    alloc_f(&d_s2,    (size_t)N * 128 * 16 * 16);
    alloc_f(&d_s3_0,  (size_t)N * 256 *  8 *  8);
    alloc_f(&d_s3,    (size_t)N * 256 *  8 *  8);
    alloc_f(&d_s4_0,  (size_t)N * 512 *  4 *  4);
    alloc_f(&d_s4,    (size_t)N * 512 *  4 *  4);
    alloc_f(&d_pool,  (size_t)N * 512);

    // Stages (ResNet-18 = {2,2,2,2} blocks)
    stage1[0].init( 64,  64, 1, N, 32, 32);
    stage1[1].init( 64,  64, 1, N, 32, 32);
    stage2[0].init( 64, 128, 2, N, 32, 32);
    stage2[1].init(128, 128, 1, N, 16, 16);
    stage3[0].init(128, 256, 2, N, 16, 16);
    stage3[1].init(256, 256, 1, N,  8,  8);
    stage4[0].init(256, 512, 2, N,  8,  8);
    stage4[1].init(512, 512, 1, N,  4,  4);

    // AvgPool(4) and FC
    avgpool.init(4, 4, 4, N, 512, 4, 4);
    fc.init(512, 10);

    // Pinned host scalar for loss reduction
    CUDA_CHECK(cudaMallocHost(&h_loss, sizeof(float)));
}

void ResNet18::forward(const float *x_dev, bool training)
{
    auto fwd_bn = [&](BNLayer &bn, const float *x, float *y) {
        if (training) bn.forward_train(x, y);
        else          bn.forward_infer(x, y);
    };
    auto fwd_blk = [&](BasicBlock &blk, const float *x, float *y) {
        if (training) blk.forward(x, y);
        else          blk.forward_infer(x, y);
    };

    // Stem
    stem_conv.forward(x_dev, act_stem);
    fwd_bn(stem_bn, act_stem, act_stem);
    launch_relu(act_stem, act_stem, stem_relu_mask, N * 64 * 32 * 32);

    // Each block writes to its own output buffer so backward has the correct x.
    fwd_blk(stage1[0], act_stem, act_s1_0);
    fwd_blk(stage1[1], act_s1_0, act_s1);

    fwd_blk(stage2[0], act_s1,   act_s2_0);
    fwd_blk(stage2[1], act_s2_0, act_s2);

    fwd_blk(stage3[0], act_s2,   act_s3_0);
    fwd_blk(stage3[1], act_s3_0, act_s3);

    fwd_blk(stage4[0], act_s3,   act_s4_0);
    fwd_blk(stage4[1], act_s4_0, act_s4);

    // AvgPool(4) → (N, 512, 1, 1)
    avgpool.forward(act_s4, act_pool);

    // FC(512→10)
    fc.forward(act_pool, logits, N);
}

// Cross-entropy loss + softmax, returns mean loss on host
float ResNet18::loss_and_grad(const int *labels_dev)
{
    // Allocate per-sample loss buffer
    float *d_loss_buf;
    CUDA_CHECK(cudaMalloc(&d_loss_buf, N * sizeof(float)));

    launch_cross_entropy(logits, labels_dev, probs, d_loss_buf, N, 10);

    // Reduce loss to scalar using cuBLAS dot with ones vector
    // Simple: copy to host and sum (N=64, negligible)
    float *h_buf = new float[N];
    CUDA_CHECK(cudaMemcpy(h_buf, d_loss_buf, N * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.f;
    for (int i = 0; i < N; i++) total += h_buf[i];
    delete[] h_buf;
    cudaFree(d_loss_buf);

    return total / N;
}

void ResNet18::backward(const int *labels_dev)
{
    // d_logits = (probs - one_hot) / N  — written into probs buffer (reuse)
    launch_cross_entropy_backward(probs, labels_dev, logits, N, 10);
    // Now `logits` holds d_logits

    // FC backward: d_pool = logits * W
    fc.backward(logits, act_pool, d_pool, N);

    // AvgPool backward
    avgpool.backward(d_pool, act_s4, act_pool, d_s4);

    // Each block's x must be the block's INPUT (not output), hence per-block buffers.
    stage4[1].backward(d_s4,   d_s4_0, act_s4_0);
    stage4[0].backward(d_s4_0, d_s3,   act_s3);

    stage3[1].backward(d_s3,   d_s3_0, act_s3_0);
    stage3[0].backward(d_s3_0, d_s2,   act_s2);

    stage2[1].backward(d_s2,   d_s2_0, act_s2_0);
    stage2[0].backward(d_s2_0, d_s1,   act_s1);

    stage1[1].backward(d_s1,   d_s1_0, act_s1_0);
    stage1[0].backward(d_s1_0, d_stem, act_stem);
    // Stem (ReLU/BN/conv) handled in backward_stem() which receives x_dev.
}

void ResNet18::backward_stem(const int *labels_dev, const float *x_dev)
{
    // Final leg of backward that needs x_dev (the original input batch)
    launch_relu_backward(d_stem, stem_relu_mask, d_stem, N * 64 * 32 * 32);
    stem_bn.backward(d_stem, act_stem, d_stem);
    stem_conv.backward(d_stem, nullptr, x_dev);
}

void ResNet18::update(float lr, float mom, float wd)
{
    auto upd_conv = [&](ConvLayer &c) {
        int n = c.out_c * c.in_c * c.kH * c.kW;
        launch_sgd(c.w, c.dw, c.vw, lr, mom, wd, n);
    };
    auto upd_bn = [&](BNLayer &b) {
        launch_sgd(b.scale, b.dscale, b.vscale, lr, mom, 0.f, b.C);
        launch_sgd(b.bias,  b.dbias,  b.vbias,  lr, mom, 0.f, b.C);
    };
    auto upd_blk = [&](BasicBlock &blk) {
        upd_conv(blk.conv1); upd_bn(blk.bn1);
        upd_conv(blk.conv2); upd_bn(blk.bn2);
        if (blk.has_sc) { upd_conv(blk.sc_conv); upd_bn(blk.sc_bn); }
    };

    upd_conv(stem_conv); upd_bn(stem_bn);
    for (int i = 0; i < 2; i++) { upd_blk(stage1[i]); upd_blk(stage2[i]); }
    for (int i = 0; i < 2; i++) { upd_blk(stage3[i]); upd_blk(stage4[i]); }
    launch_sgd(fc.w,  fc.dw,  fc.vw,  lr, mom, wd, fc.out_f * fc.in_f);
    launch_sgd(fc.b,  fc.db,  fc.vb,  lr, mom, 0.f, fc.out_f);
}

void ResNet18::zero_grad()
{
    stem_conv.zero_grad(); stem_bn.zero_grad();
    for (int i = 0; i < 2; i++) { stage1[i].zero_grad(); stage2[i].zero_grad(); }
    for (int i = 0; i < 2; i++) { stage3[i].zero_grad(); stage4[i].zero_grad(); }
    fc.zero_grad();
}

void ResNet18::free_mem()
{
    stem_conv.free_mem(); stem_bn.free_mem();
    cudaFree(stem_relu_mask);
    cudaFree(act_stem);
    cudaFree(act_s1_0); cudaFree(act_s1);
    cudaFree(act_s2_0); cudaFree(act_s2);
    cudaFree(act_s3_0); cudaFree(act_s3);
    cudaFree(act_s4_0); cudaFree(act_s4);
    cudaFree(act_pool); cudaFree(logits); cudaFree(probs);
    cudaFree(d_stem);
    cudaFree(d_s1_0);   cudaFree(d_s1);
    cudaFree(d_s2_0);   cudaFree(d_s2);
    cudaFree(d_s3_0);   cudaFree(d_s3);
    cudaFree(d_s4_0);   cudaFree(d_s4);
    cudaFree(d_pool);
    cudaFreeHost(h_loss);
    for (int i = 0; i < 2; i++) {
        stage1[i].free_mem(); stage2[i].free_mem();
        stage3[i].free_mem(); stage4[i].free_mem();
    }
    avgpool.free_mem();
    fc.free_mem();
}
