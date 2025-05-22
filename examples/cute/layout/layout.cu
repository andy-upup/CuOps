#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

using namespace cute;

TEST(Cute, MakeLayout) {
  constexpr int M = 3;
  constexpr int N = 4;
  auto l10x20 = make_layout(make_shape(Int<M>{}, Int<N>{}));
  print_layout(l10x20);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}