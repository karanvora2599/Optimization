#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

#define CUDNN_CHECK(expr) do { \
    cudnnStatus_t _s = (expr); \
    if (_s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error %s:%d  %s\n", __FILE__, __LINE__, cudnnGetErrorString(_s)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t _s = (expr); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d  code=%d\n", __FILE__, __LINE__, (int)_s); \
        exit(1); \
    } \
} while(0)
