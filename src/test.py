import torch

import flag_gems as gems

with gems.use_gems():
    x = torch.randn([2, 3], dtype=torch.float32, device=gems.device())
    y = torch.add(x, x)
    print(y)


x = torch.tensor([2, 5], device=gems.device())
with gems.device_guard(x.device):
    print(x)
