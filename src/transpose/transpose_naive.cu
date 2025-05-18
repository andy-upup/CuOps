#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>
#include <iostream>
#include <string>

namespace transpose {
__global__ void transpose_naive(float* src, float* dst, const int src_height,
                                const int src_width) {
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (row_idx >= src_height || col_idx >= src_width) {
    return;
  }
  const int src_idx = row_idx * src_width + col_idx;
  const int dst_idx = col_idx * src_height + row_idx;
  dst[dst_idx] = src[src_idx];
}

void TransposeNaive(float* input_ptr, float* output_ptr, const int src_height,
                    const int src_width) {
  dim3 block(8, 32);
  dim3 grid((src_width + block.x - 1) / block.x,
            (src_height + block.y - 1) / block.y);
  transpose_naive<<<grid, block>>>(input_ptr, output_ptr, src_height,
                                   src_width);
}

}  // namespace transpose