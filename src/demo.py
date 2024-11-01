import time

import torch

import flag_gems as gems

gems.enable()

start_time = time.time()

x = torch.randn([1024, 2048], dtype=torch.float32, device=gems.device())
x_ = torch.randn([3, 2], dtype=torch.float32, device="cuda")
y = torch.add(x, x)
z = torch.cos(x_)
print(y, z)

x = torch.tensor([2, 5], device=gems.device())
with gems.device_guard(x.device):
    c = torch.add(x, x)


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
