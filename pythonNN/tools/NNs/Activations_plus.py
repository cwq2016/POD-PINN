import torch.nn as nn
Sigmoid = nn.Sigmoid()
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
 
    def forward(self, x):
        x = x * Sigmoid(x)
        return x
    