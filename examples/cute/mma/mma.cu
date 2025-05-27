#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

#include "utils.hpp"

using namespace cute;

template <typename T, typename MMA, const int M, const int N, const int K>
__global__ void mma_simple(const T* src_a_ptr, const T* src_b_ptr,
                           T* dst_c_ptr) {
  const int idx = threadIdx.x;
  MMA tiled_mma;

  auto thr_mma = tiled_mma.get_slice(idx);

  Tensor A =
      make_tensor(make_gmem_ptr(src_a_ptr), make_shape(Int<M>{}, Int<K>{}),
                  make_stride(Int<K>{}, Int<1>{}));
  Tensor B =
      make_tensor(make_gmem_ptr(src_b_ptr), make_shape(Int<N>{}, Int<K>{}),
                  make_stride(Int<K>{}, Int<1>{}));
  Tensor C =
      make_tensor(make_gmem_ptr(dst_c_ptr), make_shape(Int<M>{}, Int<N>{}),
                  make_stride(Int<N>{}, Int<1>{}));

  auto tAgA = thr_mma.partition_A(A);
  auto tBgB = thr_mma.partition_B(B);
  auto tCgC = thr_mma.partition_C(C);

  auto tArA = thr_mma.partition_fragment_A(A);
  auto tBrB = thr_mma.partition_fragment_B(B);
  auto tCrC = thr_mma.partition_fragment_C(C);
  if (idx == 0) {
    // MMA 由MMA指令决定，不受MMAThrLayout和MMAValLayout影响
    // MMA: SM80_16x8x16_F16F16F16F16_TN
    // A,B,C 对应为: 16*16/32=8=(2,2,2), 16*8/32=4=(2,2), 16*8/32=4=(2,2)

    // MMA_M, MMA_K, MMA_N 由MMA指令、MMAThrLayout和源Tensor
    // shape决定，不受MMAValLayout影响 MMA_M = M / (mma_op_m * thr_layout_m)
    // MMA_N = N / (mma_op_n * thr_layout_n)
    // MMA_K = K / (mma_op_k * thr_layout_k)

    // (MMA, MMA_M, MMA_K)
    PRINT("tArA.shape", tArA.shape());
    // (MMA, MMA_N, MMA_K)
    PRINT("tBrB.shape", tBrB.shape());
    // (MMA, MMA_M, MMA_N)
    PRINT("tCrC.shape", tCrC.shape());
  }
  cute::copy(tAgA, tArA);
  cute::copy(tBgB, tBrB);
  clear(tCrC);

  cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  cute::copy(tCrC, tCgC);
}

template <typename T>
void gemm_naive(const T* src_a_ptr, const T* src_b_ptr, T* dst_c_ptr,
                const int M, const int N, const int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += src_a_ptr[i * K + k] * src_b_ptr[k * N + j];
      }
      dst_c_ptr[i * N + j] = (T)sum;
    }
  }
}

TEST(Cute, Mma) {
  using T = cute::half_t;
  int device;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceGetAttribute returned error %d (%s)\n", err,
            cudaGetErrorString(err));
  }
  int smem_per_block;
  err = cudaDeviceGetAttribute(&smem_per_block,
                               cudaDevAttrMaxSharedMemoryPerBlock, device);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceGetAttribute returned error %d (%s)\n", err,
            cudaGetErrorString(err));
  }

  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSetCacheConfig returned error %d (%s)\n", err,
            cudaGetErrorString(err));
  }

  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 128;
  cudaEvent_t start, end;
  float elapsed_time = 0.f;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // MMAOperation, M=16, N=8, K=16, type=half
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  // MMA_Traits
  using mma_traits = MMA_Traits<mma_op>;
  // MMA_ATOM
  using mma_atom = MMA_Atom<mma_traits>;
  // TiledMMA
  using MMA = decltype(make_tiled_mma(
      mma_atom{}, make_layout(Shape<_2, _4, _4>{}),  // thr_layout
      make_layout(Shape<_4, _4, _4>{})));            // val_layout

  T* src_a_host_ptr = (T*)malloc(M * K * sizeof(T));
  T* src_b_host_ptr = (T*)malloc(N * K * sizeof(T));
  T* dst_c_host_ptr = (T*)malloc(M * N * sizeof(T));
  T* dst_c_naive_ptr = (T*)malloc(M * N * sizeof(T));
  gen_rand_data<T>(src_a_host_ptr, M * K);
  gen_rand_data<T>(src_b_host_ptr, N * K);
  memset(dst_c_host_ptr, 0, M * N * sizeof(T));
  memset(dst_c_naive_ptr, 0, M * N * sizeof(T));

  T* src_a_dev_ptr;
  T* src_b_dev_ptr;
  T* dst_c_dev_ptr;
  cudaMalloc(&src_a_dev_ptr, M * K * sizeof(T));
  cudaMalloc(&src_b_dev_ptr, N * K * sizeof(T));
  cudaMalloc(&dst_c_dev_ptr, M * N * sizeof(T));
  dim3 block(size(MMA{}));
  cudaEventRecord(start);
  const int num_loop = 10;
  for (int i = 0; i < num_loop; ++i) {
    mma_simple<T, MMA, M, N, K>
        <<<1, block>>>(src_a_dev_ptr, src_b_dev_ptr, dst_c_dev_ptr);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  err = cudaMemcpy(dst_c_dev_ptr, dst_c_host_ptr, M * K * sizeof(T),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy (src_a) returned error %d (%s)\n", err,
            cudaGetErrorString(err));
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed with error %d (%s)\n", err,
            cudaGetErrorString(err));
  }
  cudaEventElapsedTime(&elapsed_time, start, end);
  printf("copy_global_to_shm_to_register: %f ms\n", elapsed_time / num_loop);

  gemm_naive(src_a_host_ptr, src_b_host_ptr, dst_c_naive_ptr, M, N, K);
  bool cmp = check(dst_c_host_ptr, dst_c_naive_ptr, M * N);
  if (!cmp) {
    fprintf(stderr, "Result mismatch!\n");
    for (int i = 0; i < M * N; ++i) {
      if (dst_c_host_ptr[i] != dst_c_naive_ptr[i]) {
        fprintf(stderr, "Mismatch at index %d: %f != %f\n", i,
                (float)dst_c_host_ptr[i], (float)dst_c_naive_ptr[i]);
      }
    }
  } else {
    printf("Result matches!\n");
  }

  free(src_a_host_ptr);
  free(src_b_host_ptr);
  free(dst_c_host_ptr);
  cudaFree(src_a_dev_ptr);
  cudaFree(src_b_dev_ptr);
  cudaFree(dst_c_dev_ptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}