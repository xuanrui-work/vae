import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import warnings

class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()
    
    def encode(self, x: torch.Tensor):
        raise NotImplementedError
    
    def decode(self, z: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def loss_fn(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def sample_z(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def sample(self, num_samples, device: torch.device = None) -> torch.Tensor:
        raise NotImplementedError

class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (3, 64, 64),
        hidden_dims: list[int] = None,
        latent_dim: list[int]|int = None,
        batch_norm: bool = True,
        activation: type = nn.LeakyReLU
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        if latent_dim is None:
            latent_dim = 128

        self.input_shape = input_shape
        
        conv_layers = []
        output_shape = input_shape

        for hidden_dim in hidden_dims:
            conv_layers.append(nn.Sequential(
                nn.Conv2d(output_shape[0], hidden_dim, kernel_size=3, padding='same'),
                nn.MaxPool2d(kernel_size=2),
                nn.BatchNorm2d(hidden_dim) if batch_norm else nn.Identity(),
                activation(),
            ))
            output_shape = (
                hidden_dim,
                output_shape[1]//2,
                output_shape[2]//2
            )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.output_shape = output_shape

        output_features = np.prod(output_shape)

        self.out_mu = nn.Linear(output_features, latent_dim)
        self.out_logvar = nn.Linear(output_features, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.out_mu(x)
        logvar = self.out_logvar(x)
        return (mu, logvar)

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: list[int] = None,
        reshape: tuple[int, int, int] = None,
        output_shape: tuple[int, int, int] = (3, 64, 64),
        batch_norm: bool = True,
        activation: type = nn.LeakyReLU
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512][::-1]
        if reshape is None:
            reshape = (hidden_dims[0], 2, 2)
        
        if np.prod(reshape) != latent_dim:
            warnings.warn(
                f'Mismatched dimension between reshape={reshape} and latent_dim={latent_dim}. '
                f'A nn.Linear has been prepended to the model to give the desired reshape.'
            )
            self.input_layer = nn.Linear(latent_dim, np.prod(reshape))
        else:
            self.input_layer = nn.Identity()

        self.latent_dim = latent_dim
        self.reshape = reshape

        conv_layers = []
        output_shape1 = reshape

        for hidden_dim in hidden_dims:
            conv_layers.append(nn.Sequential(
                nn.Conv2d(output_shape1[0], hidden_dim, kernel_size=3, padding=1),
                nn.Upsample(scale_factor=2),
                nn.BatchNorm2d(hidden_dim) if batch_norm else nn.Identity(),
                activation(),
            ))
            output_shape1 = (
                hidden_dim,
                output_shape1[1]*2,
                output_shape1[2]*2
            )
        
        self.conv_layers = nn.ModuleList(conv_layers)

        output_layer = [
            nn.Conv2d(hidden_dims[-1], output_shape[0], kernel_size=3, padding='same'),
            nn.Sigmoid()
        ]
        if output_shape1 != output_shape:
            output_layer.insert(
                0,
                nn.Upsample(size=output_shape[1:], mode='bilinear')
            )
            warnings.warn(
                f'Mismatch between model output shape output_shape1={output_shape1} and output_shape={output_shape}. '
                f'A nn.Upsample has been prepended to the output layer to give the desired output_shape.'
            )
        
        self.output_layer = nn.Sequential(*output_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = torch.reshape(x, (-1, *self.reshape))
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.output_layer(x)
        return x

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
        k = p_mu.shape[0]
        kl_div = 0.5*(
            torch.sum(p_var/q_var) + torch.sum((q_mu - p_mu)**2/q_var) - k + torch.sum(q_logvar - p_logvar)
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
        q_cov_inv = torch.inverse(q_cov)
        k = p_mu.shape[0]
        kl_div = 0.5*(
            torch.trace(q_cov_inv@p_cov) + (q_mu - p_mu)@q_cov_inv@(q_mu - p_mu) - k + torch.log(torch.det(q_cov)/torch.det(p_cov))
        )
        return kl_div
