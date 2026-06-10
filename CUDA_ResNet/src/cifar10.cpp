// cifar10.cpp — CIFAR-10 binary loader with CUDA pinned memory.
//
// Pinned (page-locked) host memory lets the CUDA DMA engine transfer data to
// the GPU at full PCIe bandwidth without a CPU page-locking step per transfer.
// Compared to CPP_ResNet's pageable allocations, this removes a memcpy inside
// the CUDA driver on every batch.

#include "../include/cuda_check.cuh"
#include "../include/resnet.cuh"
#include <fstream>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <random>

static const int IMAGE_BYTES = 3 * 32 * 32;  // per image, uint8
static const int RECORD      = 1 + IMAGE_BYTES;

void CIFAR10Loader::load(const char *root, bool train)
{
    // Determine which files to read
    char path[512];
    std::vector<std::string> files;
    if (train) {
        for (int i = 1; i <= 5; i++) {
            snprintf(path, sizeof(path),
                     "%s/cifar-10-batches-bin/data_batch_%d.bin", root, i);
            files.emplace_back(path);
        }
    } else {
        snprintf(path, sizeof(path), "%s/cifar-10-batches-bin/test_batch.bin", root);
        files.emplace_back(path);
    }

    // Count total images
    const size_t per_file = 10000;
    num_samples = per_file * files.size();

    // Allocate pinned host memory for float images and int labels
    CUDA_CHECK(cudaMallocHost(&h_images, num_samples * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_labels, num_samples * sizeof(int)));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_images, num_samples * 3 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_samples * sizeof(int)));

    // Read binary files
    size_t img_idx = 0;
    std::vector<uint8_t> buf(RECORD * per_file);

    for (const auto &f : files) {
        std::ifstream fin(f, std::ios::binary);
        if (!fin.is_open()) {
            fprintf(stderr, "Cannot open %s\n", f.c_str());
            continue;
        }
        fin.read(reinterpret_cast<char*>(buf.data()), buf.size());

        for (size_t i = 0; i < per_file; i++, img_idx++) {
            const uint8_t *rec = buf.data() + i * RECORD;
            h_labels[img_idx] = static_cast<int>(rec[0]);

            // Convert uint8 CHW → float CHW in [0,1]
            float *dst = h_images + img_idx * 3 * 32 * 32;
            for (int j = 0; j < IMAGE_BYTES; j++)
                dst[j] = static_cast<float>(rec[1 + j]) / 255.f;
        }
    }
}

// Copy the full dataset to device (done once at startup)
void CIFAR10Loader::to_device(cudaStream_t s)
{
    CUDA_CHECK(cudaMemcpyAsync(d_images, h_images,
        num_samples * 3 * 32 * 32 * sizeof(float),
        cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_labels, h_labels,
        num_samples * sizeof(int),
        cudaMemcpyHostToDevice, s));
}

void CIFAR10Loader::free_mem()
{
    cudaFreeHost(h_images);
    cudaFreeHost(h_labels);
    cudaFree(d_images);
    cudaFree(d_labels);
}
