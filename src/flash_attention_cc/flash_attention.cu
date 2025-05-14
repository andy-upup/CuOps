#include <assert.h>
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

void launchFlashAttentionKernel_v1(const float *__restrict__ Q,
                                   const float *__restrict__ K,
                                   const float *__restrict__ V,
                                   float *__restrict__ O, float *__restrict__ l,
                                   float *__restrict__ m, const int batch_size,
                                   const int num_head, const int N, const int M,
                                   const int d, cudaStream_t stream = 0) {
  constexpr int Bc = 4;
  assert(M % Bc == 0);
  const float softmax_scale = 1.0f / sqrtf((float)d);

  const int sram_size = (d + 2 * Bc * d + Bc) * sizeof(float);
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %g KB, requested shared memory: %g KB \n",
         max_sram_size / 1024.0f, sram_size / 1024.0f);

  constexpr int block_size = 128;
  dim3 grid_dim(num_head, batch_size);
  dim3 block_dim(block_size);
  flashAttentionKernelV1<Bc><<<grid_dim, block_dim, sram_size, stream>>>(
      Q, K, V, O, l, m, N, M, d, batch_size, num_head, softmax_scale);
}

void timingAttn(const float *Q, const float *K, const float *V,
                const int batch_size, const int num_head, const int N,
                const int M, const int d, float *l, float *m, float *O) {
  constexpr int REPEAT_NUM = 1;
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < REPEAT_NUM; ++i) {
    launchFlashAttentionKernel_v1(Q, K, V, O, l, m, batch_size, num_head, N, M,
                                  d);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float elapsed_time;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf(
      "alogrithm: flash attention v1 bz(%d) nh(%d) N(%d) M(%d) d(%d), "
      "elapsed_time: %g ms\n",
      batch_size, num_head, N, M, d, elapsed_time / REPEAT_NUM);
}

void printMatrix(const float *mat, char *s, int height, int width, int end_row,
                 int end_col, int start_row = 0, int start_col = 0) {
  assert(start_row >= 0 && start_col >= 0 && end_row <= height &&
         end_col <= width);
  printf(
      "\nmatrix %s: width=%d, height=%d, start_row=%d, end_row=%d, "
      "start_col=%d, end_col=%d\n",
      s, width, height, start_row, end_row, start_col, end_col);
  for (int i = start_row; i < end_row; i++) {
    for (int j = start_col; j < end_col; j++) {
      printf("%g\t", mat[i * width + j]);
    }
    printf("\n");
  }
}

void printVec(const float *vec, char *s, int length, int end_id,
              int start_id = 0) {
  assert(start_id >= 0 && end_id <= length);
  printf("\nvec %s: length=%d, start_id=%d, end_id=%d\n", s, length, start_id,
         end_id);
  for (int i = start_id; i < end_id; i++) {
    printf("%g\t", vec[i]);
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  constexpr int batch_size = 32;
  constexpr int num_head = 8;
  constexpr int N = 1024;
  constexpr int M = 1024;
  constexpr int d = 256;

  float *Q = new float[batch_size * num_head * N * d];
  float *K = new float[batch_size * num_head * M * d];
  float *V = new float[batch_size * num_head * M * d];
  float *l = new float[batch_size * num_head * N];
  float *m = new float[batch_size * num_head * N];
  float *O = new float[batch_size * num_head * N * d];

  // srand(1024);
  // for (int i = 0; i < batch_size * num_head * N * d; ++i)
  // {
  //     Q[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
  //     O[i] = 0.0f;
  // }

  // for (int i = 0; i < batch_size * num_head * M * d; ++i)
  // {
  //     K[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
  //     V[i] = (rand() / (RAND_MAX + 1.0f)) * 1.0f - 0.5f;
  // }

  // for (int i = 0; i < batch_size * num_head * N * d; ++i)
  // {
  //     Q[i] = i % 1003 - 500.0f;
  //     O[i] = 0.0f;
  // }

  // for (int i = 0; i < batch_size * num_head * M * d; ++i)
  // {
  //     K[i] = i % 2157 - 1218.1f;
  //     V[i] = i % 191 - 100.9f;
  // }

  // 初始化Q矩阵
  for (int i = 0; i < batch_size * num_head * N * d; ++i) {
    // Q[i] = ((i % 200) - 100) * 0.1f; // 方案一
    Q[i] = ((i * 997 % 2001) * 0.01f - 10.0f);  // 方案二
    O[i] = 0.0f;
  }

  // 初始化K矩阵（使用不同周期）
  for (int i = 0; i < batch_size * num_head * M * d; ++i) {
    K[i] = ((i % 211) - 105) * 0.095f;          // 211是质数
    V[i] = ((i * 503 % 1999) * 0.01f - 10.0f);  // 503是质数
  }

  for (int i = 0; i < batch_size * num_head * N; ++i) {
    l[i] = 0.0f;
    m[i] = -1e20f;
  }

  printMatrix(Q, (char *)("Matrix Q: "), N, d, 32, 32, 28, 24);
  printMatrix(K, (char *)("Matrix K: "), M, d, 32, 32, 28, 24);
  printMatrix(V, (char *)("Matrix V: "), M, d, 32, 32, 28, 24);

  float *d_Q;
  float *d_K;
  float *d_V;
  float *d_l;
  float *d_m;
  float *d_O;
  size_t mem_size = sizeof(float) * (batch_size * num_head * (N + M) * d * 2 +
                                     batch_size * num_head * N * 2);
  printf("requested global memory: %g GB \n",
         mem_size / 1024.0f / 1024.0f / 1024.0f);

  CHECK_CUDA_ERROR(cudaMalloc((void **)&d_Q, mem_size));
  d_K = d_Q + batch_size * num_head * N * d;
  d_V = d_K + batch_size * num_head * M * d;
  d_l = d_V + batch_size * num_head * M * d;
  d_m = d_l + batch_size * num_head * N;
  d_O = d_m + batch_size * num_head * N;

  CHECK_CUDA_ERROR(cudaMemcpy(d_Q, Q,
                              sizeof(float) * batch_size * num_head * N * d,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_K, K,
                              sizeof(float) * batch_size * num_head * M * d,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_V, V,
                              sizeof(float) * batch_size * num_head * M * d,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_l, l, sizeof(float) * batch_size * num_head * N,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_m, m, sizeof(float) * batch_size * num_head * N,
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_O, O,
                              sizeof(float) * batch_size * num_head * N * d,
                              cudaMemcpyHostToDevice));

  timingAttn(d_Q, d_K, d_V, batch_size, num_head, N, M, d, d_l, d_m, d_O);

  CHECK_CUDA_ERROR(cudaMemcpy(O, d_O,
                              sizeof(float) * batch_size * num_head * N * d,
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(l, d_l, sizeof(float) * batch_size * num_head * N,
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(m, d_m, sizeof(float) * batch_size * num_head * N,
                              cudaMemcpyDeviceToHost));

  printMatrix(O, (char *)("Matrix output: "), N, d, 32, 32, 28, 24);
  printVec(l, (char *)("Vec l: "), N, 64, 48);
  printVec(m, (char *)("Vec m: "), N, 64, 48);

  CHECK_CUDA_ERROR(cudaFree(d_Q));
  delete[] Q;
  delete[] K;
  delete[] V;
  delete[] l;
  delete[] m;
  delete[] O;

  return 0;
}