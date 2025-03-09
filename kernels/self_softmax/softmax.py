import os
import torch
import time
from torch.utils.cpp_extension import load
from functools import partial
from typing import Optional

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;9.0'

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(name='softmax_lib',
           sources=['softmax.cu'],
           extra_cuda_cflags=[
               "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ],
           extra_cflags=['-std=c++17'])


def run_benchmark(perf_func: callable, x: torch.Tensor,
                  tag: str, out: Optional[torch.Tensor] = None,
                  warmup: int = 10, iters: int = 100,
                  show_all: bool = False):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000 # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>24}: {out_val}, time:{mean_time:.8f}ms")
    if show_all: print(out)
    return out, mean_time

# per token softmax
print("-" * 100)
S, H = 1024, 1024
print(" " * 45 + f"S={S}, H={H}")
print("-" * 100)
x = torch.randn((S, H), device="cuda").cuda().float().contiguous()
out_naive = torch.zeros_like(x).cuda().float().contiguous()
out_opt = torch.zeros_like(x).cuda().float().contiguous()

run_benchmark(partial(torch.softmax, dim=1, out=out_naive), x, "f32_th(per)")
run_benchmark(lib.safe_softmax_f32_per_token,         x, "f32(safe)",        out_opt)

compare = torch.allclose(out_naive, out_opt, rtol=1e-5, atol=1e-5)
print("compare: ", compare)