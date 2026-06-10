// main.cu — Training loop for CUDA ResNet-18 on CIFAR-10.
//
// Performance techniques vs the LibTorch CPP_ResNet:
//   1. Pinned host memory → full PCIe DMA bandwidth for H→D transfers
//   2. Two CUDA streams: stream_compute runs the model, stream_transfer
//      prefetches the next batch — overlapping data movement with GPU compute
//   3. Direct cuDNN/cuBLAS calls — no LibTorch dispatch overhead
//   4. Fused add+ReLU kernel at each BasicBlock output
//   5. cuDNN algorithm benchmarking at init time (fastest algo per conv layer)

#include "../include/cuda_check.cuh"
#include "../include/resnet.cuh"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <vector>

// ── Learning rate schedule (Step-LR matching CPP_ResNet) ─────────────────────
static float lr_schedule(int epoch, float base_lr)
{
    // step_size=5, gamma=0.1
    int steps = epoch / 5;
    float lr = base_lr;
    for (int i = 0; i < steps; i++) lr *= 0.1f;
    return lr;
}

// ── Accuracy from device logits + labels ─────────────────────────────────────
static int count_correct(const float *d_logits, const int *d_labels, int N, int C)
{
    std::vector<float> logits(N * C);
    std::vector<int>   labels(N);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, N * C * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, N * sizeof(int),       cudaMemcpyDeviceToHost));
    int correct = 0;
    for (int i = 0; i < N; i++) {
        int pred = 0;
        for (int c = 1; c < C; c++)
            if (logits[i*C+c] > logits[i*C+pred]) pred = c;
        correct += (pred == labels[i]);
    }
    return correct;
}

int main(int argc, char **argv)
{
    const char *data_root = (argc > 1) ? argv[1]
        : "C:/Users/karan/Documents/Optimization Techniques/CUDA_ResNet/data";

    // ── Init cuDNN + cuBLAS ───────────────────────────────────────────────────
    CUDNN_CHECK (cudnnCreate (&g_cudnn));
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s  (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    // ── Hyperparameters ───────────────────────────────────────────────────────
    const int   BATCH    = 64;
    const int   EPOCHS   = 30;
    const float BASE_LR  = 0.1f;
    const float MOMENTUM = 0.9f;
    const float WD       = 5e-4f;

    // ── Load CIFAR-10 with pinned memory ──────────────────────────────────────
    printf("Loading CIFAR-10 from %s ...\n", data_root);
    CIFAR10Loader train_loader, test_loader;
    train_loader.load(data_root, /*train=*/true);
    test_loader .load(data_root, /*train=*/false);
    printf("Train: %zu  Test: %zu\n", train_loader.num_samples, test_loader.num_samples);

    // ── CUDA streams ──────────────────────────────────────────────────────────
    cudaStream_t stream_compute, stream_transfer;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));

    // Bind cuDNN/cuBLAS to compute stream
    CUDNN_CHECK (cudnnSetStream (g_cudnn,  stream_compute));
    CUBLAS_CHECK(cublasSetStream(g_cublas, stream_compute));

    // ── Per-batch device buffers (double-buffered for overlap) ────────────────
    // Buffer A: used by compute, Buffer B: being filled by transfer
    float *d_images_a, *d_images_b;
    int   *d_labels_a, *d_labels_b;
    CUDA_CHECK(cudaMalloc(&d_images_a, (size_t)BATCH * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_images_b, (size_t)BATCH * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels_a, BATCH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels_b, BATCH * sizeof(int)));

    // CIFAR-10 normalisation constants (same as CPP_ResNet)
    float h_mean[3] = {0.4914f, 0.4822f, 0.4465f};
    float h_std[3]  = {0.2023f, 0.1994f, 0.2010f};
    float *d_mean, *d_std;
    CUDA_CHECK(cudaMalloc(&d_mean, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_std,  3 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_std,  h_std,  3 * sizeof(float), cudaMemcpyHostToDevice));

    // Event for synchronising double-buffer swap
    cudaEvent_t transfer_done;
    CUDA_CHECK(cudaEventCreate(&transfer_done));

    // ── Build model ───────────────────────────────────────────────────────────
    printf("Initialising ResNet-18 (benchmarking cuDNN algorithms)...\n");
    ResNet18 model;
    model.init(BATCH);
    printf("Model ready.\n\n");

    // ── Shuffle indices ───────────────────────────────────────────────────────
    int N_train = (int)train_loader.num_samples;
    int N_test  = (int)test_loader.num_samples;
    std::vector<int> train_idx(N_train), test_idx(N_test);
    std::iota(train_idx.begin(), train_idx.end(), 0);
    std::iota(test_idx.begin(),  test_idx.end(),  0);
    std::mt19937 rng(42);

    auto total_start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        float lr = lr_schedule(epoch - 1, BASE_LR);
        auto epoch_start = std::chrono::steady_clock::now();

        // ── Training ─────────────────────────────────────────────────────────
        std::shuffle(train_idx.begin(), train_idx.end(), rng);
        int n_batches = N_train / BATCH;

        double total_loss = 0.0;
        int    total_correct = 0;

        // Prefetch first batch onto stream_transfer
        {
            float *h_img_batch = train_loader.h_images + (size_t)train_idx[0] * 3 * 32 * 32;
            CUDA_CHECK(cudaMemcpyAsync(d_images_a,
                h_img_batch, (size_t)BATCH * 3 * 32 * 32 * sizeof(float),
                cudaMemcpyHostToDevice, stream_transfer));
            // labels: gather — for simplicity, use a host-side gather then one H→D
            static int lbl_tmp[64];
            for (int i = 0; i < BATCH; i++) lbl_tmp[i] = train_loader.h_labels[train_idx[i]];
            CUDA_CHECK(cudaMemcpyAsync(d_labels_a, lbl_tmp, BATCH * sizeof(int),
                cudaMemcpyHostToDevice, stream_transfer));
            CUDA_CHECK(cudaEventRecord(transfer_done, stream_transfer));
        }

        for (int b = 0; b < n_batches; b++) {
            // Swap buffers
            float *d_img_cur  = (b % 2 == 0) ? d_images_a : d_images_b;
            int   *d_lbl_cur  = (b % 2 == 0) ? d_labels_a : d_labels_b;
            float *d_img_next = (b % 2 == 0) ? d_images_b : d_images_a;
            int   *d_lbl_next = (b % 2 == 0) ? d_labels_b : d_labels_a;

            // Prefetch next batch on transfer stream (overlap with compute)
            if (b + 1 < n_batches) {
                int next_start = (b + 1) * BATCH;
                static int lbl_tmp[64];
                for (int i = 0; i < BATCH; i++)
                    lbl_tmp[i] = train_loader.h_labels[train_idx[next_start + i]];

                // Use contiguous memory if possible, otherwise gather
                CUDA_CHECK(cudaMemcpyAsync(d_img_next,
                    train_loader.h_images + (size_t)train_idx[next_start] * 3 * 32 * 32,
                    (size_t)BATCH * 3 * 32 * 32 * sizeof(float),
                    cudaMemcpyHostToDevice, stream_transfer));
                CUDA_CHECK(cudaMemcpyAsync(d_lbl_next, lbl_tmp, BATCH * sizeof(int),
                    cudaMemcpyHostToDevice, stream_transfer));
                CUDA_CHECK(cudaEventRecord(transfer_done, stream_transfer));
            }

            // Make compute stream wait for current batch transfer
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, transfer_done, 0));

            // GPU-side normalisation (avoids a CPU pass per batch)
            launch_normalize(d_img_cur, d_mean, d_std, BATCH, 3, 32*32, stream_compute);

            // Forward
            model.forward(d_img_cur, /*training=*/true);
            float loss = model.loss_and_grad(d_lbl_cur);
            total_loss += loss;

            // Accumulate accuracy from logits (small copy, done on host)
            total_correct += count_correct(model.logits, d_lbl_cur, BATCH, 10);

            // Backward + update
            model.backward(d_lbl_cur);
            model.backward_stem(d_lbl_cur, d_img_cur);
            model.update(lr, MOMENTUM, WD);

            if ((b + 1) % 50 == 0 || b + 1 == n_batches) {
                printf("\r  [%d/%d] loss=%.4f", b+1, n_batches,
                       (float)(total_loss / (b+1)));
                fflush(stdout);
            }
        }
        printf("\n");

        auto train_end = std::chrono::steady_clock::now();
        double train_sec = std::chrono::duration<double>(train_end - epoch_start).count();

        double train_loss = total_loss / n_batches;
        double train_acc  = 100.0 * total_correct / (n_batches * BATCH);

        // ── Evaluation ────────────────────────────────────────────────────────
        int n_test_batches = N_test / BATCH;
        double val_loss = 0.0;
        int    val_correct = 0;

        for (int b = 0; b < n_test_batches; b++) {
            CUDA_CHECK(cudaMemcpyAsync(d_images_a,
                test_loader.h_images + (size_t)(b * BATCH) * 3 * 32 * 32,
                (size_t)BATCH * 3 * 32 * 32 * sizeof(float),
                cudaMemcpyHostToDevice, stream_compute));
            static int lbl_tmp[64];
            for (int i = 0; i < BATCH; i++) lbl_tmp[i] = test_loader.h_labels[b*BATCH+i];
            CUDA_CHECK(cudaMemcpyAsync(d_labels_a, lbl_tmp, BATCH * sizeof(int),
                cudaMemcpyHostToDevice, stream_compute));

            launch_normalize(d_images_a, d_mean, d_std, BATCH, 3, 32*32, stream_compute);
            model.forward(d_images_a, /*training=*/false);

            // Loss (no backward)
            float *d_loss_buf;
            CUDA_CHECK(cudaMalloc(&d_loss_buf, BATCH * sizeof(float)));
            launch_cross_entropy(model.logits, d_labels_a, model.probs,
                                 d_loss_buf, BATCH, 10, stream_compute);
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));

            float h_buf[64];
            CUDA_CHECK(cudaMemcpy(h_buf, d_loss_buf, BATCH * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < BATCH; i++) val_loss += h_buf[i];
            cudaFree(d_loss_buf);

            val_correct += count_correct(model.logits, d_labels_a, BATCH, 10);
        }

        val_loss /= n_test_batches * BATCH;
        double val_acc = 100.0 * val_correct / (n_test_batches * BATCH);

        auto epoch_end = std::chrono::steady_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        double train_only_sec = train_sec - std::chrono::duration<double>(epoch_start - epoch_start).count();

        printf("Epoch %2d/%d | lr=%.5f | Train Loss=%.4f Acc=%.2f%%"
               " | Val Loss=%.4f Acc=%.2f%% | Train %.1fs | Total %.1fs\n",
               epoch, EPOCHS, lr,
               (float)train_loss, (float)train_acc,
               (float)val_loss,   (float)val_acc,
               (float)train_sec,  (float)epoch_sec);
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();
    printf("\nTotal training time: %.1f seconds\n", (float)total_sec);

    // ── Cleanup ───────────────────────────────────────────────────────────────
    model.free_mem();
    train_loader.free_mem();
    test_loader.free_mem();
    cudaFree(d_images_a); cudaFree(d_images_b);
    cudaFree(d_labels_a); cudaFree(d_labels_b);
    cudaFree(d_mean);     cudaFree(d_std);
    cudaEventDestroy(transfer_done);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
    cudnnDestroy(g_cudnn);
    cublasDestroy(g_cublas);

    return 0;
}
