#include <cuda.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

#include "utils.hpp"

using namespace cute;

TEST(Cute, MakeLayout) {
  constexpr int M = 3;
  constexpr int N = 4;
  auto layout3x4 = make_layout(make_shape(Int<M>{}, Int<N>{}));
  PRINT("layout3x4", layout3x4);

  auto shape8 = make_shape(Int<8>{});
  auto stride1 = make_stride(Int<1>{});
  auto layout8_1 = make_layout(shape8, stride1);
  PRINT("layout8_1", layout8_1);

  auto shape4x5 = make_shape(Int<4>{}, Int<5>{});
  auto stride5x1 = make_stride(Int<5>{}, Int<1>{});
  auto layout4x5_5x1 = make_layout(shape4x5, stride5x1);
  PRINT("layout4x5_5x1", layout4x5_5x1);

  auto shape2x3x4 = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
  auto stride12x4x1 = make_stride(Int<12>{}, Int<4>{}, Int<1>{});
  auto layout2x3x4_12x4x1 = make_layout(shape2x3x4, stride12x4x1);
  PRINT("layout2x3x4_12x4x1", layout2x3x4_12x4x1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}