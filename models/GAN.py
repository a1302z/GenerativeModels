import torch
import torch.nn as nn
from models.common import init, Decoder, Encoder
import numpy as np

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
    
    def __init__(self, input_dim=10, num_classes=1, start_dim=128, blocks=3, output_dim=(28, 28), RGB=False):
        super(VanillaGenerator, self).__init__()
        self.output_dim = (3 if RGB else 1, *output_dim)
        if num_classes > 1:
            self.embdd = torch.nn.Embedding(num_classes, input_dim)
        else:
            self.embdd = equiv()
        modules = []
        modules.append(VanillaUpBlock(input_dim, start_dim))
        cur_dim = start_dim
        for i in range(blocks):
            modules.append(VanillaUpBlock(cur_dim, cur_dim*2))
            cur_dim *= 2
        
        self.generate = nn.Sequential(
            *modules,
            nn.Linear(cur_dim, np.prod(output_dim)),
            nn.Tanh(),
        )
        init(self)
        
        
    def forward(self, x, noise=None):
        x = self.embdd(x)
        if noise:
            x += noise
        x = self.generate(x)
        x = x.view(-1, *self.output_dim)
        return x

class VanillaDiscriminator(nn.Module):
    def __init__(self, n_classes=0, blocks=3, output_dim=(1, 28, 28), end_dim=128):
        super(VanillaDiscriminator, self).__init__()
        start_dim = end_dim * (2**(blocks-1))
        modules = [nn.Linear(np.prod(output_dim), start_dim), nn.LeakyReLU()]
        cur_dim = start_dim
        for i in range(blocks-1):
            modules.append(nn.Linear(cur_dim, cur_dim // 2))
            modules.append(nn.LeakyReLU())
            cur_dim = cur_dim // 2
        
        self.discriminate = nn.Sequential(
            *modules,
            nn.Linear(cur_dim, n_classes+1),
            nn.Sigmoid()
        )
        init(self)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.discriminate(x)

    

    
class DCGenerator(nn.Module):
    def __init__(self, input_dim=100, up_blocks=4, final_resolution=(28, 28), num_classes=0,channel_increase_factor=2,
                 conv_blocks_per_decrease=1, initial_upsample_size=3, skip_connections=False, RGB=False):
        super(DCGenerator, self).__init__()
        if num_classes > 0:
            self.embdd = torch.nn.Embedding(num_classes, input_dim)
        else:
            self.embdd = equiv()
        self.dec = Decoder(final_size=final_resolution, in_channels=input_dim, encode_factor=up_blocks, RGB = RGB,
                           channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease,
                           initial_upsample_size=initial_upsample_size, skip_connections=skip_connections, final_activation=nn.Tanh())
        init(self)
        

    def forward(self, x, noise=None):
        x = self.embdd(x)
        if noise is not None:
            x = x.view(noise.size())
            x += noise
        x = x.view(*x.size(), 1, 1)
        x = self.dec(x)
        return x
    
class DCDiscriminator(nn.Module):
    def __init__(self, channels=16, n_classes=0, up_blocks=2,channel_increase_factor=2,
                 conv_blocks_per_decrease=1, skip_connections=False, RGB=False):
        super(DCDiscriminator, self).__init__()
        self.enc = Encoder(base_channels=channels, encode_factor=up_blocks, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, RGB=RGB, skip_connections=skip_connections)
        self.classify = nn.Sequential(
            nn.Linear(channels * (channel_increase_factor**up_blocks), n_classes+1),
            nn.Sigmoid()
        )
        init(self)
        
        
    def forward(self, x):
        x = self.enc(x)
        x = x.squeeze()
        x = self.classify(x)
        return x
        
    
  
    
if __name__ == '__main__':
    from torchsummary import summary
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=['Vanilla', 'DC'], help='Which model to test?')
    parser.add_argument('--data', default='MNIST', type=str, choices=['MNIST', 'CelebA'], help='Which data format?')
    #parser.add_argument('--input_dim', type=int, default=100, help='What input dimension for generator?')
    #parser.add_argument('--aux', action='store_true', help='Test auxillary setting')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #n_classes = 10 if args.aux else 0
    
    if args.data == 'MNIST':
        input_size = (28, 28)
        RGB = False
    elif args.data == 'CelebA':
        input_size = (178, 218)
        RGB = True
    else:
        raise NotImplementedError('Dataset unkown')
    
    
    if args.model == 'Vanilla':
        print('Vanilla Generator test')
        latent_dim = 100
        encode_blocks = 3
        base_channels = 128
        
        disc_blocks=3
        
        gen = VanillaGenerator(input_dim=latent_dim, num_classes=0, start_dim=base_channels, blocks=encode_blocks, 
                                   output_dim=input_size, RGB=RGB).to(device)
        summary(gen, (latent_dim,))
        print('Vanilla Discriminator test')
        disc = VanillaDiscriminator(n_classes=0, blocks=disc_blocks, 
                                        output_dim=input_size, end_dim=base_channels).to(device)
        summary(disc, (3 if RGB else 1,*input_size))
    elif args.model == 'DC':
        latent_dim = 100
        encode_blocks = 4
        channel_increase_factor = 2
        conv_blocks_per_decrease = 8
        skip_connections = True
        initial_upsample_size = 3
        
        #DISC_PARAMS
        base_channels = 16
        disc_encode_blocks = 2
        disc_conv_blocks_per_decrease = 4
        
        print('DC Generator test')
        gen = DCGenerator(input_dim=latent_dim, up_blocks=encode_blocks, final_resolution=input_size, 
                          num_classes=0,channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease,
                          initial_upsample_size=initial_upsample_size, skip_connections=skip_connections, RGB=RGB).to(device)
        summary(gen, (latent_dim,))
        print('DC Discriminator test')
        disc = DCDiscriminator(channels=base_channels, n_classes=0, up_blocks=disc_encode_blocks,channel_increase_factor=channel_increase_factor,
                 conv_blocks_per_decrease=disc_conv_blocks_per_decrease, skip_connections=skip_connections, RGB=RGB).to(device)
        summary(disc, (3 if RGB else 1,*input_size))
    else:
        raise NotImplementedError('Model test not implemented')
    
    
"""
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
        
    def forward(self, x, noise=None):
        x = self.embdd(x)
        if noise is not None:
            x = x.view(noise.size())
            x += noise
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
"""  

