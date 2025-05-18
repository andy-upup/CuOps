#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>
#include <iostream>
#include <string>

namespace transpose {
#define FLOAT4(x) (reinterpret_cast<float4*>(&(x))[0])
template <const int kTileM, const int kTileN>
__global__ void transpose_tile(float* src, float* dst, const int src_height,
                               const int src_width) {
  const int col_idx = (blockIdx.x * blockDim.x + threadIdx.x) * kTileN;
  const int row_idx = (blockIdx.y * blockDim.y + threadIdx.y) * kTileM;
  if (row_idx >= src_height || col_idx >= src_width) {
    return;
  }
  const int src_idx = row_idx * src_width + col_idx;
  const int dst_idx = col_idx * src_height + row_idx;
  float src_tile[kTileM][kTileN];
  float dst_tile[kTileN][kTileM];
#pragma unroll
  for (int i = 0; i < kTileM; ++i) {
    FLOAT4(src_tile[i][0]) = FLOAT4(src[src_idx + i * src_width]);
  }

  FLOAT4(dst_tile[0]) = make_float4(src_tile[0][0], src_tile[1][0],
                                    src_tile[2][0], src_tile[3][0]);
  FLOAT4(dst_tile[1]) = make_float4(src_tile[0][1], src_tile[1][1],
                                    src_tile[2][1], src_tile[3][1]);
  FLOAT4(dst_tile[2]) = make_float4(src_tile[0][2], src_tile[1][2],
                                    src_tile[2][2], src_tile[3][2]);
  FLOAT4(dst_tile[3]) = make_float4(src_tile[0][3], src_tile[1][3],
                                    src_tile[2][3], src_tile[3][3]);
#pragma unroll
  for (int i = 0; i < kTileN; ++i) {
    FLOAT4(dst[dst_idx + i * src_height]) = FLOAT4(dst_tile[i][0]);
  }
}

void TransposeTile(float* input_ptr, float* output_ptr, const int src_height,
                   const int src_width) {
  constexpr int kTileM = 4;
  constexpr int kTileN = 4;
  dim3 block(16, 16);
  dim3 grid((src_width + block.x - 1) / block.x / kTileN,
            (src_height + block.y - 1) / block.y / kTileM);
  transpose_tile<kTileM, kTileN>
      <<<grid, block>>>(input_ptr, output_ptr, src_height, src_width);
}
}  // namespace transpose