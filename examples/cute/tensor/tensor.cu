#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

using namespace cute;

TEST(Cute, MakeTensor) {
  constexpr int M = 10;
  constexpr int N = 20;
  auto tensor = make_tensor<float>(make_shape(Int<M>{}, Int<N>{}));
  print(tensor);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}