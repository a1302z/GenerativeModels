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
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
            nn.LeakyReLU(),
            
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 2),
            nn.ConvTranspose2d(32, channels, 3, stride=1),
            nn.Tanh(),
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
    
    

class LinearAutoencoder(nn.Module):
    # input size being tupel of image dimension
    def __init__(self, input_size, RGB = False):
        super(LinearAutoencoder, self).__init__()
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
            nn.LeakyReLU(),
            
        )
        self.tohidden = nn.Linear((input_size[0]-7)*(input_size[1]-7)*128, 10)
        self.fromhidden = nn.Linear(10, (input_size[0]-7)*(input_size[1]-7)*128)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 2),
            nn.ConvTranspose2d(32, channels, 3, stride=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        self.s = x.size()
        x = x.view(-1)
        x = self.tohidden(x)
        hidden = x
        x = self.fromhidden(x)
        x = x.view(self.s)
        x = self.decoder(x)
        #print(x.size())
        return x
    
    def encode(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1)
        x = self.tohidden(x)
        return x
    
    def decode(self, hidden):
        x = self.fromhidden(x)
        x = x.view(self.s)
        x = self.decoder(x)
        return x