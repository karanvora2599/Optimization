cmake_minimum_required(VERSION 3.18)
project(CUDA_ResNet LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA toolkit ──────────────────────────────────────────────────────────────
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")

# ── Compiler flags ────────────────────────────────────────────────────────────
# -lineinfo  : preserves source-level profiling info (useful with Nsight)
# --use_fast_math : enables fast FP optimisations (safe for training, marginal gain)
# -arch      : generate code for your GPU SM; change if needed
#              sm_86 = RTX 30xx Ampere, sm_89 = RTX 40xx Ada, sm_80 = A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES 86)   # Change to match your GPU

# Release build for maximum performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE  "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cu
    src/kernels.cu
    src/layers.cu
    src/resnet.cu
    src/cifar10.cpp
)

add_executable(CUDA_ResNet ${SOURCES})

# ── cuDNN + cuBLAS ────────────────────────────────────────────────────────────
# These ship with the CUDA toolkit installer; link them directly.
target_link_libraries(CUDA_ResNet
    cuda
    cudart
    cudnn
    cublas
)

# ── Post-build: copy runtime DLLs (Windows only) ──────────────────────────────
if(MSVC)
    # cudnn and cublas DLLs must be next to the .exe
    foreach(dll cudnn64_9 cublas64_12 cublasLt64_12)
        find_file(DLL_${dll} "${dll}.dll"
            PATHS "${CUDA_TOOLKIT_ROOT_DIR}/bin"
            NO_DEFAULT_PATH)
        if(DLL_${dll})
            add_custom_command(TARGET CUDA_ResNet POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${DLL_${dll}}" "$<TARGET_FILE_DIR:CUDA_ResNet>")
        endif()
    endforeach()
endif()

message(STATUS "Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch   : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
