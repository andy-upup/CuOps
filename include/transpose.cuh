#ifndef TRANSPOSE_CUH_
#define TRANSPOSE_CUH_
namespace transpose {
void TransposeNaive(float* input_ptr, float* output_ptr, const int src_height,
                    const int src_width);

void TransposeSmem(float* input_ptr, float* output_ptr, const int src_height,
                   const int src_width);

void TransposeTile(float* input_ptr, float* output_ptr, const int src_height,
                   const int src_width);
}  // namespace transpose
#endif