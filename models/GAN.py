import torch
import torch.nn as nn

class VanillaUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VanillaUpBlock, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.fwd(x)

"""
Currently hardcoded for MNIST
"""
class VanillaGenerator(nn.Module):
    
    def __init__(self, input_dim=10):
        super(VanillaGenerator, self).__init__()
        self.generate = nn.Sequential(
            VanillaUpBlock(input_dim, 128),
            VanillaUpBlock(128, 256),
            VanillaUpBlock(256, 512),
            VanillaUpBlock(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
        
        
    def forward(self, x):
        x = self.generate(x)
        x = x.view(-1, 1, 28, 28)
        return x
    
class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super(VanillaDiscriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, stride=1),
        )
        self.classify = nn.Linear(512, 1)
        self.sm = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.discriminate(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return self.sm(x)
    
    
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Generator test')
    gen = VanillaGenerator().to(device)
    summary(gen, (10,))
    
    #exit()
    
    print('Discriminator test')
    disc = VanillaDiscriminator().to('cuda')
    summary(disc, (1,28,28))
    