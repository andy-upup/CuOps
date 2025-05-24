#ifndef UTILS_HPP_
#define UTILS_HPP_
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
bool check(const T* output, const T* golden, const int size) {
  for (int i = 0; i < size; ++i) {
    if (std::is_same<T, int>::value) {
      if (output[i] != golden[i]) {
        return false;
      }
    } else if (std::is_same<T, float>::value) {
      if (std::abs(output[i] - golden[i]) >= 1e-4) {
        return false;
      }
    } else if (std::is_same<T, double>::value) {
      if (std::abs(output[i] - golden[i]) >= 1e-8) {
        return false;
      }
    }
  }
  return true;
}

#endif  // UTILS_HPP_