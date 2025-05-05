#include <float.h>
#include <torch/torch.h>

#include <iostream>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

bool check(std::vector<float>& output, std::vector<float>& golden,
           const int N) {
  for (int i = 0; i < N; ++i) {
    if (std::abs(output[i] - golden[i]) >= 1e-4) {
      return false;
    }
  }
  return true;
}

void safe_softmax_naive(const std::vector<float>& input,
                        std::vector<float>& output) {
  float max_val = *std::max_element(input.begin(), input.end());
  float sum = 0.0f;

  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }

  for (size_t i = 0; i < output.size(); ++i) {
    output[i] /= sum;
  }
}

void safe_online_softmax(const std::vector<float>& input,
                         std::vector<float>& output) {
  float max_val = -FLT_MAX;
  float last_max_val = 0.f;
  float sum = 0.f;

  for (size_t i = 0; i < input.size(); ++i) {
    max_val = std::max(max_val, input[i]);
    sum = sum * std::exp(last_max_val - max_val) + std::exp(input[i] - max_val);
    last_max_val = max_val;
  }

  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = std::exp(input[i] - max_val) / sum;
  }
}

float safe_online_softmax_dot_product(const std::vector<float>& input,
                                      const std::vector<float>& value) {
  float max_val = -FLT_MAX;
  float last_max_val = 0.f;
  float sum = 0.f;
  float result = 0.f;

  for (size_t i = 0; i < input.size(); ++i) {
    max_val = std::max(max_val, input[i]);
    sum = sum * std::exp(last_max_val - max_val) + std::exp(input[i] - max_val);
    last_max_val = max_val;
  }

  for (size_t i = 0; i < input.size(); ++i) {
    result += std::exp(input[i] - max_val) / sum * value[i];
  }
  return result;
}

float safe_online_softmax_dot_product_one_loop(
    const std::vector<float>& input, const std::vector<float>& value) {
  float max_val = -FLT_MAX;
  float last_max_val = 0.f;
  float sum = 0.f;
  float last_sum = 0.f;
  float last_result = 1.f;
  float result = 0.f;

  for (size_t i = 0; i < input.size(); ++i) {
    max_val = std::max(max_val, input[i]);
    sum = last_sum * std::exp(last_max_val - max_val) +
          std::exp(input[i] - max_val);
    result = last_result * std::exp(last_max_val - max_val) * last_sum / sum +
             std::exp(input[i] - max_val) / sum * value[i];
    last_result = result;
    last_sum = sum;
    last_max_val = max_val;
  }

  // for (size_t i = 0; i < input.size(); ++i) {
  //   result += std::exp(input[i] - max_val) / sum * value[i];
  // }
  return result;
}

int main() {
  // const int batch_size = 16;
  // const int n_head = 12;
  // const int seq_len = 64;
  // const int head_embd = 64;

  // auto q = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
  // auto k = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
  // auto v = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();

  // forward(q, k, v);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> value = {0.1f, 0.2f, 0.3f, 0.4f};
  // std::vector<float> output(input.size());
  // std::vector<float> golden(input.size());
  // safe_softmax_naive(input, golden);
  // safe_online_softmax(input, output);
  // if (check(output, golden, input.size())) {
  //   std::cout << "safe_online_softmax passed!" << std::endl;
  // } else {
  //   std::cout << "safe_online_softmax failed!" << std::endl;
  // }

  float result = safe_online_softmax_dot_product(input, value);
  float golden = safe_online_softmax_dot_product_one_loop(input, value);
  if (std::abs(result - golden) < 1e-4) {
    std::cout << "safe_online_softmax_dot_product passed!" << std::endl;
  } else {
    std::cout << "safe_online_softmax_dot_product failed!" << std::endl;
  }
  return 0;
}