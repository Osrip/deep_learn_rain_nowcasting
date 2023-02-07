import torch
from modules_blocks import MetDilBlock, MetResModule, SampleModule
# device = torch.device('cuda')
device = torch.device('cpu')

test_tensor = torch.randn((3, 3, 32, 32), device=device)
# test_tensor = torch.randn((1, 1, 3, 3), device=device)

# model = MetDilBlock(3, 32, 2, 3).to(device)
model = MetResModule(c_num=3, width_height=32, kernel_size=3).to(device)
# model = SampleModule(c_in=3, c_out=2*3, width_height_out=32/2).to(device)
x = model(test_tensor)
pass