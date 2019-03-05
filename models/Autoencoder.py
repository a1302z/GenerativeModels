import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        conv1 = nn.Conv2d(3, 32, 3)
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(32, 64, 3)
        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(64, 128, 2)
        
        decv3 = nn.ConvTranspose2d(128, 64, 2)
        upol2 = nn.MaxUnpool2d(2)
        decv2 = nn.ConvTranspose2d(64, 32, 3)
        upol1 = nn.MaxUnpool2d(2)
        decv1 = nn.ConvTranspose2d(32, 3, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        hidden = x
        x = self.decv3(x)
        x = self.upol2(x)
        x = self.decv2(x)
        x = self.upol1(x)
        x = self.decv1(x)
        return x
