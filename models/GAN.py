import torch
import torch.nn as nn

class GAN(nn.Module):
    
    def __init__(self, noise_dim=2, img_dim=(1,28,28)):
        super(GAN, self).__init__()
        self.noise_dim = noise_dim
        self.img_dim = img_dim
        self.out_dim = img_dim[0]*img_dim[1]*img_dim[2]
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 16),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32,128),
            nn.LeakyReLU(),
            nn.Linear(128,512),
            nn.LeakyReLU(),
            nn.Linear(512, self.out_dim),
            nn.Sigmoid()
        )
        self.discriminator = nn.Sequential(
            nn.Linear(self.out_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.view(x.size()[0],-1)
        print(x.size())
        x = self.generator(x)
        x = self.discriminator(x)
        return x
    
    def generate(self, n_samples, cuda=False):
        x = torch.randn((n_samples, self.noise_dim), requires_grad = False)
        if cuda:
            x = x.cuda()
        x = self.generator(x)
        return x.view(-1, *self.img_dim)
    
    
    def discriminate(self, x):
        x = x.view(x.size()[0], -1)
        x = self.discriminator(x)
        return x
    
    
 
            
    
            