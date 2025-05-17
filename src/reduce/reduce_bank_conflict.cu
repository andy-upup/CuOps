#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>

namespace reduce {
#define THREAD_PER_BLOCK 256

__global__ void reduce_bank_conflict(const float* src, float* dst,
                                     const int N) {
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[THREAD_PER_BLOCK];
  shared[tx] = src[tid];
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

void ReduceBankConflict(const float* input_ptr, float* output_ptr,
                        const int num) {
  const int num_block = num / THREAD_PER_BLOCK;
  dim3 grid(num_block);
  dim3 block(THREAD_PER_BLOCK);
  reduce_bank_conflict<<<grid, block>>>(input_ptr, output_ptr, num);
}

}  // namespace reduce