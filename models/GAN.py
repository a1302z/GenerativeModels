import torch
import torch.nn as nn

class VanillaGenerator(nn.Module):
    
    def __init__(self, noise_dim=2, RGB=False):
        super(VanillaGenerator, self).__init__()
        self.RGB = RGB
        factor = 1
        if RGB:
            factor = 3
        
        self.create_start = nn.Linear(noise_dim, factor*(noise_dim**2))
        self.create_img = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 64, 2),
        )
            