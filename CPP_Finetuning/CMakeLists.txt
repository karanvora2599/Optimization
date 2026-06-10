cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPT2FineTuner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── LibTorch ──────────────────────────────────────────────────────────────────
# Set CMAKE_PREFIX_PATH to your LibTorch install dir, e.g.:
#   cmake .. -DCMAKE_PREFIX_PATH="C:/libtorch"
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ─── nlohmann/json (header-only, fetched automatically) ────────────────────────
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
    DOWNLOAD_NO_EXTRACT TRUE
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/nlohmann
)
FetchContent_MakeAvailable(nlohmann_json)

# ─── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/model/gpt2.cpp
    src/lora/lora_linear.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/data/dataset.cpp
    src/training/trainer.cpp
    src/logging/logger.cpp
)

add_executable(finetuner ${SOURCES})

target_include_directories(finetuner PRIVATE
    include
    third_party
)

target_link_libraries(finetuner PRIVATE "${TORCH_LIBRARIES}")
target_compile_features(finetuner PRIVATE cxx_std_17)

# ─── Windows: copy LibTorch DLLs next to the executable ───────────────────────
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET finetuner POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS} $<TARGET_FILE_DIR:finetuner>
    )
endif()
