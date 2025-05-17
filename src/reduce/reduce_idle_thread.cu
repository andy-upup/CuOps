#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>

#define THREAD_PER_BLOCK 256

__global__ void reduce_idle_thread(const float* src, float* dst, const int N) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int tid = (blockIdx.x * blockDim.x) * 2 + threadIdx.x;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[tx] = src[tid] + src[tid + THREAD_PER_BLOCK];
  __syncthreads();
  for (int i = THREAD_PER_BLOCK / 2; i >= 1; i /= 2) {
    if (tx < i) {
      shared[tx] += shared[tx + i];
    }
    __syncthreads();
  }
  if (tx == 0) {
    dst[bx] = shared[0];
  }
}

// bool check(const float* output, const float* golden, const int N) {
//   for (int i = 0; i < N; ++i) {
//     if (std::abs(output[i] - golden[i]) >= 1e-4) {
//       return false;
//     }
//   }
//   return true;
// }

// int main() {
//   const int N = 32 * 1024 * 1024;
//   float* input = (float*)malloc(N * sizeof(float));
//   float* d_input;
//   cudaMalloc((void**)&d_input, N * sizeof(float));

//   const int num_block = N / THREAD_PER_BLOCK / 2;
//   float* output = (float*)malloc(num_block * sizeof(float));
//   float* d_output;
//   cudaMalloc((void**)&d_output, num_block * sizeof(float));
//   float* golden = (float*)malloc(num_block * sizeof(float));

//   for (int i = 0; i < N; ++i) {
//     input[i] = 2.0 * (float)drand48() - 1.0;
//   }

//   for (int i = 0; i < num_block; ++i) {
//     float sum_block = 0.f;
//     for (int j = 0; j < THREAD_PER_BLOCK * 2; ++j) {
//       sum_block += input[i * THREAD_PER_BLOCK * 2 + j];
//     }
//     golden[i] = sum_block;
//   }

//   cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

//   dim3 grid(num_block);
//   dim3 block(THREAD_PER_BLOCK);
//   reduce_idle_thread<<<grid, block>>>(d_input, d_output, N);

//   cudaMemcpy(output, d_output, num_block * sizeof(float),
//              cudaMemcpyDeviceToHost);
//   if (check(output, golden, num_block)) {
//     printf("Output is right.\n");
//   } else {
//     printf("Output is wrong!\n");
//     for (int i = 0; i < num_block; ++i) {
//       printf("%lf", output[i]);
//     }
//     printf("\n");
//   }

//   cudaFree(d_input);
//   cudaFree(d_output);
//   free(output);
//   free(golden);
//   return 0;
// }