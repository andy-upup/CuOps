#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>
#include <iostream>
#include <string>

namespace transpose {
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
template <const int kBlockSize>
__global__ void transpose_smem(float* src, float* dst, const int src_height,
                               const int src_width) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idy >= src_height || idx >= src_width) {
    return;
  }
  __shared__ float smem_tile[kBlockSize][kBlockSize];
  float* src_start_block = src + by * kBlockSize * src_width + bx * kBlockSize;
  smem_tile[tx][ty] = src_start_block[ty * src_width + tx];
  __syncthreads();

  float* dst_start_block = dst + bx * kBlockSize * src_height + by * kBlockSize;
  dst_start_block[ty * src_height + tx] = smem_tile[ty][tx];
}

void TransposeSmem(float* input_ptr, float* output_ptr, const int src_height,
                   const int src_width) {
  constexpr int kTileSize = 16;
  dim3 block(16, 16);
  dim3 grid((src_width + block.x - 1) / block.x,
            (src_height + block.y - 1) / block.y);
  transpose_smem<kTileSize>
      <<<grid, block>>>(input_ptr, output_ptr, src_height, src_width);
}
}  // namespace transpose