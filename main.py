import torch
from manager_gpu import GPUManager
manager = GPUManager()
torch.cuda.set_device(manager.auto_choice())
