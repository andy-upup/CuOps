#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math_constants.h>
#include <stdio.h>

#define CHECK_CUBLAS_STATUS(status)               \
  do {                                            \
    cublasStatus_t err = (status);                \
    if (err != CUBLAS_STATUS_SUCCESS) {           \
      fprintf(stderr, "CUBLAS error: %d\n", err); \
      exit(EXIT_FAILURE);                         \
    }                                             \
  } while (0)

#define CHECK_CUDA_ERROR(err)                                            \
  do {                                                                   \
    cudaError_t err_code = (err);                                        \
    if (err_code != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err_code)); \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

template <typename T>
__device__ T blockAllReduceMax(T val) {
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

template <typename T>
__device__ T blockAllReduceSum(T val) {
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void softmaxKernel(const float *__restrict__ mat,
                              float *__restrict__ output, const int ncol,
                              const float softmax_scale) {
  float val;
  float vmax = -FLT_MAX;
  float exp_sum = 1e-10f;

#pragma unroll
  for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
    vmax = max(mat[blockIdx.x * ncol + i], vmax);
  }
  __syncthreads();

  vmax = blockAllReduceMax<float>(vmax);

#pragma unroll
  for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
    exp_sum += __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale);
  }
  __syncthreads();

  exp_sum = blockAllReduceSum<float>(exp_sum);

#pragma unroll
  for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
    val = __expf((mat[blockIdx.x * ncol + i] - vmax) * softmax_scale) / exp_sum;
    output[blockIdx.x * ncol + i] = val;
  }
}

void launchSoftmaxKernel(const float *__restrict__ mat,
                         float *__restrict__ output, const int ncol,
                         const int nrow, const float softmax_scale,
                         cudaStream_t stream) {
  constexpr int block_size = 256;
  dim3 block(block_size);
  dim3 grid(nrow);
  softmaxKernel<<<grid, block, 0, stream>>>(mat, output, ncol, softmax_scale);
}

void launchAttentionBaseline(const float *__restrict__ Q,
                             const float *__restrict__ K,
                             const float *__restrict__ V,
                             float *__restrict__ QK,
                             float *__restrict__ QK_softmax,
                             float *__restrict__ O, const int batch_size,
                             const int num_head, const int N, const int M,
                             const int d, cudaStream_t stream) {
  const float softmax_scale = 1.0f / sqrtf((float)d);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  CHECK_CUBLAS_STATUS(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, d, &alpha, K, d, M * d, Q, d,
      N * d, &beta, QK, M, N * M, batch_size * num_head));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  launchSoftmaxKernel(QK, QK_softmax, M, batch_size * num_head * N,
                      softmax_scale, stream);
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_CUBLAS_STATUS(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, M, &alpha, V, d, M * d,
      QK_softmax, M, N * M, &beta, O, d, N * d, batch_size * num_head));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}