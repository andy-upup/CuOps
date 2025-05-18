#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cstdio>
#include <iostream>
#include <string>

#include "perf.hpp"
#include "transpose.cuh"
#include "utils.hpp"

TEST(Transpose, TransposeNaive) {
  constexpr int src_height = 2048;
  constexpr int src_width = 512;
  constexpr int num_iter = 5;

  constexpr int num_data = src_height * src_width;
  float* input = (float*)malloc(num_data * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, num_data * sizeof(float));

  float* output = (float*)malloc(num_data * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_data * sizeof(float));
  float* golden = (float*)malloc(num_data * sizeof(float));

  for (int i = 0; i < num_data; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < src_height; ++i) {
    for (int j = 0; j < src_width; ++j) {
      golden[j * src_height + i] = input[i * src_width + j];
    }
  }

  cudaMemcpy(d_input, input, num_data * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < num_iter; ++i) {
    perf::Perf perf("transpose_8x32");
    transpose::TransposeNaive(d_input, d_output, src_height, src_width);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(output, d_output, num_data * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_data)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_data; ++i) {
      printf("%lf", output[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

TEST(Transpose, TransposeSmem) {
  constexpr int src_height = 2048;
  constexpr int src_width = 512;
  constexpr int num_iter = 5;

  constexpr int num_data = src_height * src_width;
  float* input = (float*)malloc(num_data * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, num_data * sizeof(float));

  float* output = (float*)malloc(num_data * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_data * sizeof(float));
  float* golden = (float*)malloc(num_data * sizeof(float));

  for (int i = 0; i < num_data; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < src_height; ++i) {
    for (int j = 0; j < src_width; ++j) {
      golden[j * src_height + i] = input[i * src_width + j];
    }
  }

  cudaMemcpy(d_input, input, num_data * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < num_iter; ++i) {
    perf::Perf perf("transpose_smem");
    transpose::TransposeSmem(d_input, d_output, src_height, src_width);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(output, d_output, num_data * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_data)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_data; ++i) {
      printf("%f, %f", output[i], golden[i]);
    }
    printf("\n");
  }

  cudaFree(d_input);
  cudaFree(d_output);
  free(output);
  free(golden);
}

TEST(Transpose, TransposeTile) {
  constexpr int src_height = 2048;
  constexpr int src_width = 512;
  constexpr int num_iter = 5;

  constexpr int num_data = src_height * src_width;
  float* input = (float*)malloc(num_data * sizeof(float));
  float* d_input;
  cudaMalloc((void**)&d_input, num_data * sizeof(float));

  float* output = (float*)malloc(num_data * sizeof(float));
  float* d_output;
  cudaMalloc((void**)&d_output, num_data * sizeof(float));
  float* golden = (float*)malloc(num_data * sizeof(float));

  for (int i = 0; i < num_data; ++i) {
    input[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < src_height; ++i) {
    for (int j = 0; j < src_width; ++j) {
      golden[j * src_height + i] = input[i * src_width + j];
    }
  }

  cudaMemcpy(d_input, input, num_data * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < num_iter; ++i) {
    perf::Perf perf("transpose_tile4x4");
    transpose::TransposeTile(d_input, d_output, src_height, src_width);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(output, d_output, num_data * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (check(output, golden, num_data)) {
    printf("Output is right.\n");
  } else {
    printf("Output is wrong!\n");
    for (int i = 0; i < num_data; ++i) {
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