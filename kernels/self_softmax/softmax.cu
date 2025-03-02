#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <algorithm>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  value = warp_reduce_sum_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];

  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}

// dim3 block(128);
// dim3 grid((col + block.x - 1) / block.x, row);
template <const int NUM_THREADS = 1024>
__global__ void softmax_f32_naive_kernel(float* src, float* dst, const int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // col
  const int idy = blockIdx.y;                             // row

  if (idx >= NUM_THREADS) {
    return;
  }
  float row_max = (float)src[idy * NUM_THREADS];
  for (int i = 0; i < NUM_THREADS; ++i) {
    row_max = fmaxf(row_max, src[idy * NUM_THREADS + i]);
  }

  float row_sum = 0.f;
  for (int i = 0; i < NUM_THREADS; ++i) {
    row_sum += expf(src[idy * NUM_THREADS + i] - row_max);
  }

  dst[idy * NUM_THREADS + idx] =
      (expf(src[idy * NUM_THREADS + idx] - row_max) / row_sum);
}

// safe_softmax per token
template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;

  float val = (idx < N) ? x[idx] : -FLT_MAX;
  float max_val = block_reduce_max_f32<NUM_THREADS>(val);  // block max
  float exp_val = (idx < N) ? expf(x[idx] - max_val) : 0.0f;
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);  // block sum
  // e^x_i/sum(e^x_0,...,e^x_n-1)
  if (idx < N) y[idx] = exp_val / exp_sum;
}

// --------------------- PyTorch bindings for custom kernel
// -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                   \
  if (((T).options().dtype() != (th_type))) {                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                 \
  assert((T1).dim() == (T2).dim());                      \
  for (int i = 0; i < (T1).dim(); ++i) {                 \
    if ((T2).size(i) != (T1).size(i)) {                  \
      throw std::runtime_error("Tensor size mismatch!"); \
    }                                                    \
  }

// softmax naive
#define LANUCH_SOFTMAX_F32_NAIVE_KERNEL(col)                    \
  softmax_f32_naive_kernel<(col)>                               \
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                        reinterpret_cast<float*>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_NAIVE_KERNEL(row, col)                      \
  dim3 block((128));                                                     \
  dim3 grid((((col) + block.x - 1) / (block.x)), (row));                 \
  switch ((col)) {                                                       \
    case 32:                                                             \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(32)                                \
      break;                                                             \
    case 64:                                                             \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(64)                                \
      break;                                                             \
    case 128:                                                            \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(128)                               \
      break;                                                             \
    case 256:                                                            \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(256)                               \
      break;                                                             \
    case 512:                                                            \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(512)                               \
      break;                                                             \
    case 1024:                                                           \
      LANUCH_SOFTMAX_F32_NAIVE_KERNEL(1024)                              \
      break;                                                             \
    default:                                                             \
      throw std::runtime_error("only support col: 64/128/256/512/1024"); \
      break;                                                             \
  }

// safe softmax per token
#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(col)           \
  safe_softmax_f32_per_token_kernel<(col)>                      \
      <<<grid, block>>>(reinterpret_cast<float*>(x.data_ptr()), \
                        reinterpret_cast<float*>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(row, col)             \
  dim3 block((col));                                                     \
  dim3 grid((row));                                                      \
  switch ((col)) {                                                       \
    case 32:                                                             \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32)                       \
      break;                                                             \
    case 64:                                                             \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64)                       \
      break;                                                             \
    case 128:                                                            \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128)                      \
      break;                                                             \
    case 256:                                                            \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256)                      \
      break;                                                             \
    case 512:                                                            \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512)                      \
      break;                                                             \
    case 1024:                                                           \
      LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024)                     \
      break;                                                             \
    default:                                                             \
      throw std::runtime_error("only support col: 64/128/256/512/1024"); \
      break;                                                             \
  }

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int row = x.size(0);  // row
  const int col = x.size(1);  // col
  const int N = row * col;
  DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(row, col)
}

void softmax_f32_naive(torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int row = x.size(0);  // row
  const int col = x.size(1);  // col
  const int N = row * col;
  DISPATCH_SOFTMAX_F32_NAIVE_KERNEL(row, col)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
  TORCH_BINDING_COMMON_EXTENSION(softmax_f32_naive)
}
