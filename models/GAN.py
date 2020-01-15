import torch
import torch.nn as nn
from models.common import init
#from common import init

class equiv(nn.Module):
    def __init__(self):
        super(equiv, self).__init__()
        
    def forward(self, x):
        return x


class VanillaUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VanillaUpBlock, self).__init__()
        self.fwd = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
        )
        init(self)

    def forward(self, x):
        return self.fwd(x)

"""
Currently hardcoded for MNIST
"""
class VanillaGenerator(nn.Module):
    
    def __init__(self, input_dim=10, num_classes=1):
        super(VanillaGenerator, self).__init__()
        if num_classes > 1:
            self.embdd = torch.nn.Embedding(num_classes, input_dim)
        else:
            self.embdd = equiv()
        self.generate = nn.Sequential(
            VanillaUpBlock(input_dim, 128),
            VanillaUpBlock(128, 256),
            VanillaUpBlock(256, 512),
            VanillaUpBlock(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
        init(self)
        
        
    def forward(self, x):
        x = self.embdd(x)
        x = self.generate(x)
        x = x.view(-1, 1, 28, 28)
        return x

class VanillaDiscriminator(nn.Module):
    def __init__(self, n_classes=0):
        super(VanillaDiscriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, n_classes+1),
            nn.Sigmoid()
        )
        init(self)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminate(x)
    
class DCGenerator(nn.Module):
    def __init__(self, input_dim=100, channels=128, final_resolution=28, num_classes=1):
        super(DCGenerator, self).__init__()
        if num_classes > 1:
            self.embdd = torch.nn.Embedding(num_classes, input_dim)
        else:
            self.embdd = equiv()
        self.start_res = final_resolution // 4
        self.channels = channels
        self.to_img = nn.Linear(input_dim, self.channels*self.start_res**2)
        self.generate = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Dropout(),
            nn.Conv2d(self.channels, self.channels//2, 3),
            nn.BatchNorm2d(self.channels//2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels//2, self.channels//2, 3),
            nn.LeakyReLU(),
            nn.PixelShuffle(2),
            nn.ConvTranspose2d(self.channels//8, self.channels//8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(self.channels//8, 1, 3),
            nn.Upsample(size=(final_resolution,final_resolution)),
            nn.Tanh()
        )
        init(self)
        
    def forward(self, x):
        x = self.embdd(x)
        x = self.to_img(x)
        x = x.view(x.size(0), self.channels, self.start_res, self.start_res)
        x = self.generate(x)
        return x
    
class DCDiscriminator(nn.Module):
    def __init__(self, channels=128, n_classes=0):
        super(DCDiscriminator, self).__init__()
        self.discriminate = nn.Sequential(
            nn.Conv2d(1, channels, 3),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(channels, channels*2, 3),
            nn.BatchNorm2d(channels*2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(),
            nn.Conv2d(channels*2, channels*4, 2),
            nn.BatchNorm2d(channels*4),
            nn.LeakyReLU(),
            nn.AvgPool2d(3, stride=1),
        )
        self.classify = nn.Linear(channels*16, n_classes+1)
        self.sm = nn.Sigmoid()
        init(self)
        
        
    def forward(self, x):
        x = self.discriminate(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return self.sm(x)
    
    
if __name__ == '__main__':
    from torchsummary import summary
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=['Vanilla', 'DC'], help='Which model to test?')
    parser.add_argument('--input_dim', type=int, default=100, help='What input dimension for generator?')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'Vanilla':
        print('Vanilla Generator test')
        gen = VanillaGenerator(input_dim=args.input_dim).to(device)
        summary(gen, (args.input_dim,))
        print('Vanilla Discriminator test')
        disc = VanillaDiscriminator().to('cuda')
        summary(disc, (1,28,28))
    elif args.model == 'DC':
        print('DC Generator test')
        gen = DCGenerator(input_dim=args.input_dim).to(device)
        summary(gen, (args.input_dim,))
        print('DC Discriminator test')
        disc = DCDiscriminator().to('cuda')
        summary(disc, (1,28,28))
    else:
        raise NotImplementedError('Model test not implemented')
    

