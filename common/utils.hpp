#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <cuda.h>

#include <cute/tensor.hpp>
#include <iostream>

#define PRINT(name, content) \
  print(name);               \
  print(" : ");              \
  print(content);            \
  print("\n");

#define PRINT_TENSOR(name, content) \
  print(name);                      \
  print(" : ");                     \
  print_tensor(content);            \
  print("\n");

template <typename T>
bool check(const T *output, const T *golden, const int size) {
  for (int i = 0; i < size; ++i) {
    if (std::is_same<T, int>::value) {
      if (output[i] != golden[i]) {
        return false;
      }
    } else {
      if (std::abs((float)output[i] - (float)golden[i]) >= 1e-4) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
void gen_rand_data(T *data, const int size) {
  for (int i = 0; i < size; ++i) {
    if (std::is_same<T, int>::value) {
      data[i] = rand() % 100;
    } else if (std::is_same<T, float>::value) {
      data[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    } else if (std::is_same<T, double>::value) {
      data[i] = static_cast<double>(rand()) / RAND_MAX * 100.0;
    }
  }
}

template <typename T>
__global__ void gpu_compare_kernel(const T *x, const T *y, int n,
                                   float threshold, int *count,
                                   float *max_error) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  float v0 = x[idx];
  float v1 = y[idx];

  float diff = fabs(v0 - v1);
  if (diff > threshold) {
    atomicAdd(count, 1);

    // for positive floating point, there int representation is in the same
    // order.
    int int_diff = *((int *)(&diff));
    atomicMax((int *)max_error, int_diff);
  }
}

#endif  // UTILS_HPP_