#ifndef REDUCE_CUH_
#define REDUCE_CUH_
namespace reduce {
void ReduceBankConflict(const float* input_ptr, float* output_ptr,
                        const int num);

void ReduceIdleThread(const float* input_ptr, float* output_ptr, const int num);

void ReduceSharedMemory(const float* input_ptr, float* output_ptr,
                        const int num);

void ReduceWarpDivergence(const float* input_ptr, float* output_ptr,
                          const int num);
}  // namespace reduce
#endif