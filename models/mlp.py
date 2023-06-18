import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3, hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float32) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU()])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float32)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.lin(x)