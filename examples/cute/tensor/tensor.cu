#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

#include "utils.hpp"

using namespace cute;

TEST(Cute, StackTensor) {
  auto shape = make_shape(Int<4>{}, Int<3>{});
  auto stride = make_stride(Int<3>{}, Int<1>{});
  auto tensor = make_tensor<float>(shape, stride);
  PRINT("tensor_layout", tensor.layout());
  PRINT("tensor_shape", tensor.shape());
  PRINT("tensor_stride", tensor.stride());
  PRINT("tensor_size", tensor.size());
  PRINT("tensor_data", tensor.data());
}

TEST(Cute, HeapTensor) {
  using T = float;
  constexpr int kDataSize = 12;
  T* h_data_ptr = (T*)malloc(kDataSize * sizeof(T));

  /*
  - - - - - -
  | 0  1  2 |
  | 3  4  5 |
  | 6  7  8 |
  | 9  10 11|
  - - - - - -
  */
  auto shape = make_shape(Int<4>{}, Int<3>{});
  auto stride = make_stride(Int<3>{}, Int<1>{});
  // host data pointer
  auto tensor = make_tensor(make_gmem_ptr(h_data_ptr), shape, stride);
  PRINT("tensor_layout", tensor.layout());
  PRINT("tensor_shape", tensor.shape());
  PRINT("tensor_stride", tensor.stride());
  PRINT("tensor_size", tensor.size());
  PRINT("tensor_data", tensor.data());
  PRINT_TENSOR("initial tensor", tensor);

  auto coord = make_coord(2, 1);
  PRINT("tensor(2, 1)", tensor(coord));

  auto tensor_slice = tensor(_, 1);
  PRINT_TENSOR("tensor_slice", tensor_slice);

  // split the tensor into 2x3 tiles of size 2x1, and select the tile at
  // position (0, 1).
  auto tensor_tile = local_tile(tensor, make_shape(2, 1), make_coord(0, 1));
  PRINT_TENSOR("tensor_tile", tensor_tile);

  // split the tensor into 2x3 tiles of size 2x1, and take th 0th element from
  // each tile.
  int idx = 0;
  auto tensor_partition = local_partition(
      tensor, make_layout(make_shape(2, 1), make_stride(1, 1)), idx);
  PRINT_TENSOR("tensor_partition", tensor_partition);
  free(h_data_ptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}