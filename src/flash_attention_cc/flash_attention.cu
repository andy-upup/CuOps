#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math_constants.h>
#include <stdio.h>

#include <iostream>

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

struct __align__(8) MD_F {
  float m;  // max val
  float d;  // exp sum
};

struct MDFOp {
  __device__ __forceinline__ MD_F operator()(MD_F &a, MD_F &b) {
    MD_F ret;
    ret.m = max(a.m, b.m);
    ret.d = a.d * __expf(a.m - ret.m) + b.d * __expf(b.m - ret.m);
    return ret;
  }
};

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

// block_size = (128)
// grid_size = (num_head, batch_size)
// Q: [batch_size, num_head, N, d]
// K: [batch_size, num_head, M, d]
// V: [batch_size, num_head, M, d]
// O: [batch_size, num_head, N, d]
// l: [batch_size, num_head, N]
// m: [batch_size, num_head, N]
template <int Bc>
__global__ void flashAttentionKernelV1(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ O, float *__restrict__ l,
    float *__restrict__ m, const int N, const int M, const int d,
    const int batch_size, const int num_head, const float softmax_scale) {
  const int idx = threadIdx.x;
  const int block_size = blockDim.x;
  const int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
  const int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * M * d;
  const int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

  extern __shared__ float shared_mem[];
  float *q_shared = shared_mem;          // [1, d]
  float *k_shared = q_shared + d;        // [Bc, d]
  float *v_shared = k_shared + Bc * d;   // [Bc, d]
  float *qk_shared = v_shared + Bc * d;  // [1, Bc]

  __shared__ MD_F row_ml_prev;
  for (int i = 0; i < M; i += Bc) {
    for (int j = idx; j < Bc * d; j += block_size) {
      k_shared[j] = K[kv_offset + i * d + j];
      v_shared[j] = V[kv_offset + i * d + j];
    }
    __syncthreads();

    for (int j = 0; j < N; ++j) {
      for (int k = idx; k < d; k += block_size) {
        q_shared[k] = Q[qo_offset + j * d + k];
      }
      if (idx == 0) {
        row_ml_prev = {m[lm_offset + j], l[lm_offset + j]};
      }
      __syncthreads();
      MD_F row_ml = {-1e20f, 0.f};
      for (int k = 0; k < Bc; ++k) {
        MD_F tmp_ml = {0.f, 1.f};
        // 每个线程计算d / block_size个mla
        for (int x = idx; x < d; x += block_size) {
          tmp_ml.m += q_shared[x] * k_shared[k * Bc + x];
        }
        tmp_ml.m *= softmax_scale;
        __syncthreads();
        // TODO: complete blockAllReduceSum and blockAllReduceMax
        // 每个线程计算完，block内做reduce，得到Q_dot_K的一个最终结果
        tmp_ml.m = blockAllReduceSum<float>(tmp_ml.m);
        // 更新Bc段内的m和l
        row_ml = MDFOp()(tmp_ml, row_ml);
        // Bc段内的所有Q_dot_K值存储到qk_shared中
        if (idx == 0) {
          qk_shared[k] = tmpl_ml.m;
        }
        __syncthreads();
      }
      __syncthreads();
      // 每一个Bc段（长度为Bc）计算完之后更新所在行的m和l
      MD_F row_ml_new = MDFOp()(row_ml_prev, row_ml);
      for (int k = idx; k < d; k += block_size) {
        float pv = 0.f;
        for (int x = 0; x < Bc; ++x) {
          pv += __expf(qk_shared[x] - row_ml.m) * v_shared[x * d + k];
        }
        O[qo_offset + j * d + k] =
            1.0f / row_ml_new.d *
            (row_ml_prev.d * __expf(row_ml_prev.m - row_ml_new.m) *
                 O[qo_offset + j * d + k] +
             __expf(row_ml.m - row_ml_new.m) * pv);
      }
      if (idx == 0) {
        l[lm_offset + j] = row_ml_new.d;
        m[lm_offset + j] = row_ml_new.m;
      }
      __syncthreads();
    }
    __syncthreads();
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