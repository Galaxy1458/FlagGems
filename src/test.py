import time

import torch

import flag_gems as gems

# start_time = time.time()  # 记录开始时间

# 需要测量的代码段


gems.enable(vendor="nvidia")
# with gems.use_gems():
#     x = torch.randn([2, 3], dtype=torch.float32, device='cuda')
#     x_ = torch.randn([3, 2], dtype=torch.float32, device="cuda")
#     # y = torch.add(x, x)
#     z = torch.mm(x, x_)
#     print(z)

# x = torch.tensor([2, 5], device=gems.device())
# with gems.device_guard(x.device):
#     print(x)
start_time = time.time()  # 记录开始时间

x = torch.randn([2, 3], dtype=torch.float32, device="cuda")
x_ = torch.randn([3, 2], dtype=torch.float32, device="cuda")
y = torch.add(x, x)
z = torch.sin(x_)
print(y, z)


end_time = time.time()  # 记录结束时间

execution_time = end_time - start_time  # 计算执行时间
print(f"Execution time: {execution_time:.6f} seconds")
