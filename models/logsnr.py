import torch
import torch.nn as nn
import torch.nn.functional as F

class LogSNR(nn.Module):
    def __init__(self):
        super().__init__()
        self.PosDense_1 = PosDense(in_channels=1, out_channels=1)
        self.PosDense_2 = PosDense(in_channels=1, out_channels=1024)
        self.PosDense_3 = PosDense(in_channels=1024, out_channels=1)
        self.gamma_min = nn.Parameter(torch.tensor([-10.]))
        self.gamma_gap = nn.Parameter(torch.tensor([20.]))

    def forward(self, t):
        t = torch.cat([torch.tensor([0., 1.], device=t.device), t], dim=0)
        l1 = self.PosDense_1(t[:, None])
        l2 = F.sigmoid(self.PosDense_2(l1))
        schedule = torch.squeeze(l1 + self.PosDense_3(l2), dim=-1)
        s0, s1, sched = schedule[0], schedule[1], schedule[2:]
        norm_nlogsnr = (sched - s0) / (s1 - s0)
        nlogsnr = self.gamma_min + F.softplus(self.gamma_gap) * norm_nlogsnr
        return - nlogsnr


class PosDense(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kernel = nn.Parameter(torch.zeros((in_channels, out_channels)))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.xavier_normal_(self.kernel)

    def forward(self, t):
        return torch.matmul(t, F.softplus(self.kernel)) + F.softplus(self.bias)
