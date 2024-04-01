import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class KLDivDiagonalGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        p_mu: torch.Tensor, p_var: torch.Tensor,
        q_mu: torch.Tensor, q_var: torch.Tensor
    ):
        p_logvar = torch.log(p_var)
        q_logvar = torch.log(q_var)
        k = p_mu.shape[-1]
        kl_div = 0.5*(
            torch.sum(p_var/q_var, dim=-1) +
            torch.sum((q_mu - p_mu)**2/q_var, dim=-1) +
            torch.sum(q_logvar - p_logvar, dim=-1) - k
        )
        return kl_div

class KLDivFullGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        p_mu: torch.Tensor, p_cov: torch.Tensor,
        q_mu: torch.Tensor, q_cov: torch.Tensor
    ):
        q_cov_inv = torch.inverse(q_cov)        # (n, k, k)
        diff = (q_mu - p_mu).unsqueeze(-1)      # (n, k, 1)
        diff_T = diff.transpose(-1, -2)         # (n, 1, k)
        k = p_mu.shape[-1]
        kl_div = 0.5*(
            torch.vmap(torch.trace)(q_cov_inv@p_cov) +
            (diff_T@q_cov_inv@diff).squeeze((-1, -2)) +
            torch.log(torch.det(q_cov)/torch.det(p_cov)) - k
        )
        return kl_div

class EntropyDiagonalGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        mu: torch.Tensor, var: torch.Tensor
    ):
        logvar = torch.log(var)
        k = var.shape[-1]
        h = 0.5*(
            torch.sum(logvar, dim=-1) +
            k*(1 + torch.log(torch.tensor(2*np.pi, device=var.device)))
        )
        return h

class EntropyFullGaussian(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        mu: torch.Tensor, cov: torch.Tensor
    ):
        k = mu.shape[-1]
        h = 0.5*(
            torch.log(torch.det(cov)) +
            k*(1 + torch.log(torch.tensor(2*np.pi, device=cov.device)))
        )
        return h
