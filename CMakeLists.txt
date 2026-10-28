cmake_minimum_required(VERSION 3.16)
project(llm_inference_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Baseline flags. Note: no -ffast-math, so the naive fp32 path stays bit-comparable
# across compilers; later SIMD kernels opt in to their own instruction sets.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

add_library(engine_core
  src/tokenizer/tokenizer.cpp
  src/ops/ops.cpp
  src/ops/kernels.cpp
  src/ops/thread_pool.cpp
  src/model/model.cpp
  src/sampler.cpp
)
target_include_directories(engine_core PUBLIC src)
target_link_libraries(engine_core PUBLIC Threads::Threads)

# SIMD backends: each file gets only its own ISA flags; the choice is made at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_sources(engine_core PRIVATE src/ops/kernels_avx2.cpp)
  set_source_files_properties(src/ops/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  target_sources(engine_core PRIVATE src/ops/kernels_neon.cpp)
  # Apple clang already targets M1+ (which has SDOT); elsewhere ask for ARMv8.2 dotprod explicitly.
  if(NOT APPLE)
    check_cxx_compiler_flag("-march=armv8.2-a+dotprod" HAS_DOTPROD)
    if(HAS_DOTPROD)
      set_source_files_properties(src/ops/kernels_neon.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    endif()
  endif()
endif()

add_executable(engine src/main.cpp)
target_link_libraries(engine PRIVATE engine_core)

add_executable(tokenize tests/tokenize_cli.cpp)
target_link_libraries(tokenize PRIVATE engine_core)

# Unit tests (GoogleTest is a test-only dependency; the engine itself has none).
option(ENGINE_BUILD_TESTS "Build unit tests" OFF)
if(ENGINE_BUILD_TESTS)
  find_package(GTest QUIET)  # use a system install (e.g. brew install googletest) if present
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_executable(test_ops tests/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE engine_core GTest::gtest_main)
  target_compile_definitions(test_ops PRIVATE OP_VECTORS_PATH="${CMAKE_SOURCE_DIR}/tests/data/op_vectors.bin")
  include(GoogleTest)
  gtest_discover_tests(test_ops)
endif()

add_executable(bench_kernels bench/bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE engine_core)

add_executable(server src/server/server.cpp)
target_link_libraries(server PRIVATE engine_core)
