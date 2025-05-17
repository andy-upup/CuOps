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

// bool check(const float* output, const float* golden, const int N) {
//   for (int i = 0; i < N; ++i) {
//     if (std::abs(output[i] - golden[i]) >= 1e-4) {
//       return false;
//     }
//   }
//   return true;
// }

// class Perf {
//  public:
//   Perf(const std::string& name) {
//     name_ = name;
//     cudaEventCreate(&start_);
//     cudaEventCreate(&end_);
//     cudaEventRecord(start_);
//     cudaEventRecord(end_);
//   }
//   ~Perf() {
//     cudaEventRecord(end_);
//     cudaEventSynchronize(end_);
//     float duration = 0.f;
//     cudaEventElapsedTime(&duration, start_, end_);
//     std::cout << name_ << " duration: " << duration * 1000 << " us"
//               << std::endl;
//   }

//  private:
//   std::string name_;
//   cudaEvent_t start_;
//   cudaEvent_t end_;
// };

// int main() {
//   constexpr int src_height = 2048;
//   constexpr int src_width = 512;
//   constexpr int num_iter = 5;
//   constexpr int kTileM = 4;
//   constexpr int kTileN = 4;

//   constexpr int num_data = src_height * src_width;
//   float* input = (float*)malloc(num_data * sizeof(float));
//   float* d_input;
//   cudaMalloc((void**)&d_input, num_data * sizeof(float));

//   float* output = (float*)malloc(num_data * sizeof(float));
//   float* d_output;
//   cudaMalloc((void**)&d_output, num_data * sizeof(float));
//   float* golden = (float*)malloc(num_data * sizeof(float));

//   for (int i = 0; i < num_data; ++i) {
//     input[i] = 2.0 * (float)drand48() - 1.0;
//   }

//   for (int i = 0; i < src_height; ++i) {
//     for (int j = 0; j < src_width; ++j) {
//       golden[j * src_height + i] = input[i * src_width + j];
//     }
//   }

//   cudaMemcpy(d_input, input, num_data * sizeof(float),
//   cudaMemcpyHostToDevice);

//   dim3 block(16, 16);
//   dim3 grid((src_width + block.x - 1) / block.x / kTileN,
//             (src_height + block.y - 1) / block.y / kTileM);
//   for (int i = 0; i < num_iter; ++i) {
//     Perf perf("transpose_tile4x4");
//     transpose_tile<kTileM, kTileN>
//         <<<grid, block>>>(d_input, d_output, src_height, src_width);
//     cudaDeviceSynchronize();
//   }

//   cudaMemcpy(output, d_output, num_data * sizeof(float),
//              cudaMemcpyDeviceToHost);
//   if (check(output, golden, num_data)) {
//     printf("Output is right.\n");
//   } else {
//     printf("Output is wrong!\n");
//     for (int i = 0; i < num_data; ++i) {
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