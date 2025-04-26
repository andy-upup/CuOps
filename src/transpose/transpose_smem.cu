#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include <cstdio>
#include <iostream>
#include <string>

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

bool check(const float* output, const float* golden, const int N) {
  for (int i = 0; i < N; ++i) {
    if (std::abs(output[i] - golden[i]) >= 1e-4) {
      return false;
    }
  }
  return true;
}

class Perf {
 public:
  Perf(const std::string& name) {
    name_ = name;
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
    cudaEventRecord(start_);
    cudaEventRecord(end_);
  }
  ~Perf() {
    cudaEventRecord(end_);
    cudaEventSynchronize(end_);
    float duration = 0.f;
    cudaEventElapsedTime(&duration, start_, end_);
    std::cout << name_ << " duration: " << duration * 1000 << " us"
              << std::endl;
  }

 private:
  std::string name_;
  cudaEvent_t start_;
  cudaEvent_t end_;
};

int main() {
  constexpr int src_height = 2048;
  constexpr int src_width = 512;
  constexpr int num_iter = 5;
  constexpr int kTileSize = 16;

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

  dim3 block(16, 16);
  dim3 grid((src_width + block.x - 1) / block.x,
            (src_height + block.y - 1) / block.y);
  for (int i = 0; i < num_iter; ++i) {
    Perf perf("transpose_smem");
    transpose_smem<kTileSize>
        <<<grid, block>>>(d_input, d_output, src_height, src_width);
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
  return 0;
}