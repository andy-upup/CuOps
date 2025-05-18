#ifndef PERF_HPP_
#define PERF_HPP_
#include <cuda.h>

#include <iostream>
#include <string>

namespace perf {
class Perf {
 public:
  Perf(const std::string& name) {
    name_ = name;
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
    cudaEventRecord(start_);
    cudaEventRecord(end_);
  }
  ~Perf() {
    cudaEventRecord(end_);
    cudaEventSynchronize(end_);
    float duration = 0.f;
    cudaEventElapsedTime(&duration, start_, end_);
    std::cout << name_ << " duration: " << duration * 1000 << " us"
              << std::endl;
  }

 private:
  std::string name_;
  cudaEvent_t start_;
  cudaEvent_t end_;
};

}  // namespace perf

#endif  // PERF_HPP_