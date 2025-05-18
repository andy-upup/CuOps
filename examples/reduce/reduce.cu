#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cstdio>

#include "perf.hpp"
#include "reduce.cuh"
#include "utils.hpp"

#define THREAD_PER_BLOCK 256

TEST(Reduce, ReduceBankConflict) {
  const int N = 32 * 1024 * 1024;
  float* input = (float*)malloc(N * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, N * sizeof(float));

  const int num_block = N / THREAD_PER_BLOCK;
  float* output = (float*)malloc(num_block * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_block * sizeof(float));
  float* golden = (float*)malloc(num_block * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < num_block; ++i) {
    float sum_block = 0.f;
    for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
      sum_block += input[i * THREAD_PER_BLOCK + j];
    }
    golden[i] = sum_block;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  {
    perf::Perf perf("reduce_bank_conflict");
    reduce::ReduceBankConflict(d_input, d_output, N);
  }

  cudaMemcpy(output, d_output, num_block * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_block)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_block; ++i) {
      printf("%lf", output[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

TEST(Reduce, ReduceIdleThread) {
  const int N = 32 * 1024 * 1024;
  float* input = (float*)malloc(N * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, N * sizeof(float));

  const int num_block = N / THREAD_PER_BLOCK / 2;
  float* output = (float*)malloc(num_block * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_block * sizeof(float));
  float* golden = (float*)malloc(num_block * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < num_block; ++i) {
    float sum_block = 0.f;
    for (int j = 0; j < THREAD_PER_BLOCK * 2; ++j) {
      sum_block += input[i * THREAD_PER_BLOCK * 2 + j];
    }
    golden[i] = sum_block;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  {
    perf::Perf perf("reduce_idle_thread");
    reduce::ReduceIdleThread(d_input, d_output, N);
  }

  cudaMemcpy(output, d_output, num_block * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_block)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_block; ++i) {
      printf("%lf", output[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

TEST(Reduce, ReduceSharedMemory) {
  const int N = 32 * 1024 * 1024;
  float* input = (float*)malloc(N * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, N * sizeof(float));

  const int num_block = N / THREAD_PER_BLOCK;
  float* output = (float*)malloc(num_block * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_block * sizeof(float));
  float* golden = (float*)malloc(num_block * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < num_block; ++i) {
    float sum_block = 0.f;
    for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
      sum_block += input[i * THREAD_PER_BLOCK + j];
    }
    golden[i] = sum_block;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

  {
    perf::Perf perf("reduce_shared_memory");
    reduce::ReduceSharedMemory(d_input, d_output, N);
  }

  reduce::ReduceSharedMemory(d_input, d_output, N);

  cudaMemcpy(output, d_output, num_block * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_block)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_block; ++i) {
      printf("%lf", output[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

TEST(Reduce, ReduceWarpDivergence) {
  const int N = 32 * 1024 * 1024;
  float* input = (float*)malloc(N * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, N * sizeof(float));

  const int num_block = N / THREAD_PER_BLOCK;
  float* output = (float*)malloc(num_block * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_block * sizeof(float));
  float* golden = (float*)malloc(num_block * sizeof(float));

  for (int i = 0; i < N; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < num_block; ++i) {
    float sum_block = 0.f;
    for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
      sum_block += input[i * THREAD_PER_BLOCK + j];
    }
    golden[i] = sum_block;
  }

  cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
  {
    perf::Perf perf("reduce_warp_divergence");
    reduce::ReduceWarpDivergence(d_input, d_output, N);
  }

  cudaMemcpy(output, d_output, num_block * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_block)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_block; ++i) {
      printf("%lf", output[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}