import sys
sys.path.append("../advanced_VT/")
from models.models import *
from torch import nn
from Fid_computer import Kid_Or_Fid
import time

g_fine = nn.DataParallel(fine_generator()).cuda()
g_coarse = nn.DataParallel(coarse_generator()).cuda()

g_fine.module.load_state_dict(torch.load('weights/exp6/g_model_fine.pt'))
g_coarse.module.load_state_dict(torch.load('weights/exp6/g_model_coarse.pt'))

computer = Kid_Or_Fid(if_cuda=False)
start_time = time.time()
computer.update_models(g_fine_model=g_fine, g_coarse_model=g_coarse)
spin_result = computer.spin_once()
print(spin_result)
print(type(spin_result[0]))
print(type(spin_result[1]))
end_time = time.time()
print('Consume time:', end_time-start_time)