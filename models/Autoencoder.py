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
            nn.Sigmoid(),
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
    def __init__(self, input_size, RGB = False, hidden_size=(512, 128)):
        super(LinearAutoencoder, self).__init__()
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
        )
        self.tohidden = nn.Sequential(
            nn.Linear((input_size[0]-7)*(input_size[1]-7)*128, hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
        )
        self.fromhidden = nn.Sequential(
            nn.Linear(hidden_size[1], (input_size[0]-7)*(input_size[1]-7)*128),
            nn.LeakyReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 2),
            nn.ConvTranspose2d(32, channels, 3, stride=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        self.s = x.size()
        x = x.view(self.s[0], -1)
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
        self.s = x.size()
        x = x.view(self.s[0], -1)
        x = self.tohidden(x)
        return x
    
    def decode(self, hidden):
        x = self.fromhidden(x)
        x = x.view(self.s)
        x = self.decoder(x)
        return x
    
class VariationalAutoencoder(nn.Module):
    # input size being tupel of image dimension
    def __init__(self, input_size, hidden_size=(128,15), RGB = False):
        super(VariationalAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
            nn.LeakyReLU(),
            
        )
        self.hidden = nn.Linear((input_size[0]-7)*(input_size[1]-7)*128, hidden_size[0])
        self.mean = nn.Linear(hidden_size[0], hidden_size[1])
        self.log_var = nn.Linear(hidden_size[0], hidden_size[1])
        
        self.fromhidden = nn.Sequential(
            nn.Linear(hidden_size[1], hidden_size[0]),
            nn.Linear(hidden_size[0], (input_size[0]-7)*(input_size[1]-7)*128)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, channels, 3, stride=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        self.s = x.size()
        x = x.view(self.s[0], -1)
        x = self.hidden(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        x = self.sample_z(mean, log_var)
        
        x = self.fromhidden(x)
        x = x.view(self.s)
        x = self.decoder(x)
        return x, mean, log_var
    
    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(self.hidden_size[1], requires_grad=True)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu + torch.exp(log_var / 2) * eps
        

    
    def encode(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        self.s = x.size()
        x = x.view(self.s[0], -1)
        x = self.hidden(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
    
    def decode(self, sample_vector):
        x = self.fromhidden(sample_vector)
        x = x.view(self.s)
        x = self.decoder(x)
        return x