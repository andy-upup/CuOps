#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

#include "utils.hpp"

using namespace cute;

template <typename T, typename G2SCopy, typename S2RCopy, typename SmemLayout,
          const int M, const int N>
__global__ void copy_global_to_shm_to_register(const T* src_ptr) {
  const int idx = threadIdx.x;
  extern __shared__ T smem[];
  T* smem_ptr = smem;

  // row first
  auto gA = make_tensor(make_gmem_ptr(src_ptr), make_shape(Int<M>{}, Int<N>{}),
                        make_stride(Int<N>{}, Int<1>{}));
  auto sA = make_tensor(make_smem_ptr(smem_ptr), SmemLayout{});
  auto rA = make_tensor_like(gA);

  // global to shared
  G2SCopy g2s_tiled_copy;
  auto g2s_thr_copy = g2s_tiled_copy.get_slice(idx);
  auto tAgA = g2s_thr_copy.partition_S(gA);
  auto tAsA = g2s_thr_copy.partition_D(sA);
  cute::copy(g2s_tiled_copy, tAgA, tAsA);

  // shared to register
  S2RCopy s2r_tiled_copy;
  auto s2r_thr_copy = s2r_tiled_copy.get_slice(idx);
  auto stAsA = s2r_thr_copy.retile_S(tAsA);
  auto tArA = s2r_thr_copy.partition_D(rA);
  cute::copy(s2r_tiled_copy, stAsA, tArA);
}

TEST(Cute, G2S2RCopy) {
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
  cudaEvent_t start, end;
  float elapsed_time = 0.f;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  // 1. global to shared memory
  /*
   Use cp.async to copy data from global memory to shared memory,
   128bits each thread.
  */
  // CopyOperation
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  // Copy_Traits
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  // Copy_Atom
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  // TildCopy
  using G2SCopy =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<32>{}),
                                      make_stride(Int<32>{}, Int<1>{}))));
  using SmemLayout =
      decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<M>{}, Int<N>{})));

  static int smem_size = cute::cosize(SmemLayout{}) * sizeof(T);

  // 2. shared memory to registers
  /*
    The ldmatrix instruction enables warp-level data movement from shared memory
    to registers. It allows a single thread to handle 16 bytes of data transfer,
    and all threads within a warp can collectively transfer up to 512 bytes. A
    single ldmatrix instruction can load a 16x16 float16 matrix.
  */
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopy =
      decltype(make_tiled_copy(s2r_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  T* src_ptr;
  cudaMalloc(&src_ptr, M * N * sizeof(T));
  dim3 block(128);
  cudaEventRecord(start);
  const int num_loop = 10;
  for (int i = 0; i < num_loop; ++i) {
    copy_global_to_shm_to_register<T, G2SCopy, S2RCopy, SmemLayout, M, N>
        <<<1, block, smem_size>>>(src_ptr);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed with error %d (%s)\n", err,
            cudaGetErrorString(err));
  }
  cudaEventElapsedTime(&elapsed_time, start, end);
  printf("copy_global_to_shm_to_register: %f ms\n", elapsed_time / num_loop);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}