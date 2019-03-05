import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, RGB = False):
        super(Autoencoder, self).__init__()
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2),
            nn.ConvTranspose2d(64, 64, 2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
                nn.ConvTranspose2d(32, 32, 2),
            nn.ConvTranspose2d(32, 3, 3, stride=1),
        )
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        hidden = x
        #print(x.size())
        x = self.decoder(x)
        #print(x.size())
        return x
