import torch
from torch import nn

# GPU로 돌아가게 하기
device = (
    "cuda"
    if torch.cuda.is_available() 
    else "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)


