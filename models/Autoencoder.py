import torch
import torch.nn as nn
from models.common import init

class SkipConnection(nn.Module):
    def __init__(self, module=lambda x: x):
        super(SkipConnection, self).__init__()
        self.module = module
        
    def forward(self, x):
        return self.module(x)+x



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(ConvBlock, self).__init__()
        out = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        if bn:
            out.append(nn.BatchNorm2d(out_channels))
        out.append(nn.LeakyReLU())
        self.enc = nn.Sequential(*out)
        
    def forward(self, x):
        y = self.enc(x)
        #if x.size() == y.size():
        #    y += x
        return y
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(ConvTransposeBlock, self).__init__()
        out = [nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)]
        if bn:
            out.append(nn.BatchNorm2d(out_channels))
        out.append(nn.LeakyReLU())
        self.dec = nn.Sequential(*out)
        
    def forward(self, x):
        y = self.dec(x)
        #if y.size() == x.size():
        #    y += x
        return y

class Encoder(nn.Module):
    def __init__(self, base_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, RGB=False, skip_connections=False):
        super(Encoder, self).__init__()
        encode_modules = [ConvBlock(3 if RGB else 1, base_channels, bn = False)]
        cur_channels = base_channels
        for i in range(encode_factor):
            skip_modules = []
            for j in range(conv_blocks_per_decrease):
                skip_modules.append(SkipConnection(ConvBlock(cur_channels, cur_channels, bn = False)))
            if skip_connections and conv_blocks_per_decrease > 0:
                encode_modules.append(nn.Sequential(*skip_modules))
            else:
                encode_modules.extend(skip_modules)
            encode_modules.append(ConvBlock(cur_channels, cur_channels*channel_increase_factor, bn = True))
            cur_channels *= channel_increase_factor
            encode_modules.append(nn.MaxPool2d(2))
        self.enc = nn.Sequential(
            *encode_modules,
            nn.Conv2d(cur_channels, cur_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
    def forward(self, x):
        return self.enc(x)
    
class Decoder(nn.Module):    
    def __init__(self, final_size=(28,28), in_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, RGB=False, initial_upsample_size=3, skip_connections=False):
        super(Decoder, self).__init__()
        decode_modules = []
        cur_channels = in_channels
        for i in range(encode_factor):
            decode_modules.append(nn.Upsample(scale_factor=initial_upsample_size) if i == 0 else nn.Upsample(scale_factor=2))
            skip_modules = []
            for j in range(conv_blocks_per_decrease):
                skip_modules.append(SkipConnection(ConvTransposeBlock(cur_channels, cur_channels, bn = False)))
            if skip_connections and conv_blocks_per_decrease > 0:
                decode_modules.append(SkipConnection(nn.Sequential(*skip_modules)))
            else:
                decode_modules.extend(skip_modules)
            decode_modules.append(ConvTransposeBlock(cur_channels, cur_channels//channel_increase_factor, bn=True))
            cur_channels = cur_channels//channel_increase_factor
        out_channels = 3 if RGB else 1
        decode_modules.append(nn.Upsample(final_size, mode='bilinear', align_corners=False))
        skip_modules = []
        for j in range(conv_blocks_per_decrease):
            skip_modules.append(SkipConnection(ConvTransposeBlock(cur_channels, cur_channels, bn = False)))
        if skip_connections and conv_blocks_per_decrease > 0:
            decode_modules.append(SkipConnection(nn.Sequential(*skip_modules)))
        else:
            decode_modules.extend(skip_modules)
        decode_modules.append(nn.Conv2d(cur_channels, out_channels, 3, padding=1))
        decode_modules.append(nn.Sigmoid())
        
        
        self.dec = nn.Sequential(*decode_modules)
        
    
    def forward(self, x):
        return self.dec(x)
    
    
class Autoencoder(nn.Module):
    def __init__(self, variational=False, final_size=(28,28), base_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, encoding_dimension=128, RGB=False, initial_upsample_size=3, skip_connections=False):
        super(Autoencoder, self).__init__()
        self.variational = variational
        self.encoding_dimension = encoding_dimension
        self.encoder = Encoder(base_channels=base_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, RGB=RGB, skip_connections=skip_connections)
        self.final_channels = base_channels * (channel_increase_factor**encode_factor)
        self.decoder = Decoder(final_size=final_size, in_channels=self.final_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, RGB=RGB, initial_upsample_size=initial_upsample_size, skip_connections=skip_connections)
        self.to_hidden = nn.Linear(self.final_channels, encoding_dimension)
        if self.variational:
            self.log_var = nn.Linear(self.final_channels, encoding_dimension)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.from_hidden = nn.Linear(encoding_dimension, self.final_channels)
        
    def forward(self, x):
        x = self.encoder(x)
        s = x.size()
        x = x.view(x.size(0), -1)
        if self.variational:
            m = self.to_hidden(x)
            log_var = self.log_var(x)
            x = self.sample_z(m, log_var)
        else:
            x = self.to_hidden(x)
        x = self.from_hidden(x)
        x = x.view(s)
        x = self.decoder(x)
        if self.variational:
            return x, m, log_var
        else:
            return x
    
    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(mu.size(), requires_grad=True, device=self.device)
        return mu + torch.exp(log_var / 2) * eps


"""
Implementation of loss function to weight losses according to Alex Kendalls paper:
"Multi-Task Learning Using Uncertainty to Weigh Lossesfor Scene Geometry and Semantics"
"""
class WeightedMultiLoss(nn.Module):
    def __init__(self, init_values=[1.0,1.0], learn_weights=True):
        super(WeightedMultiLoss, self).__init__()
        self.weight0 = nn.Parameter(torch.tensor(init_values[0], dtype=torch.float), requires_grad=learn_weights)
        self.weight1 = nn.Parameter(torch.tensor(init_values[1], dtype=torch.float), requires_grad=learn_weights)
        
    def forward(self, x):
        l0 = (x[0] / torch.exp(2 * self.weight0) + self.weight0)
        l1 = (x[1] / torch.exp(2 * self.weight1) + self.weight1)
        return [l0, l1]

    
if __name__ == '__main__':
    from torchsummary import summary
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=['AE', 'VAE', 'test'], help='Which model to test?')
    parser.add_argument('--data', type=str, choices=['MNIST', 'CelebA'], default='MNIST', help='Which dataset shape to use')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for summary estimation')
    #parser.add_argument('--input_dim', type=int, default=100, help='What input dimension for generator?')
    #parser.add_argument('--aux', action='store_true', help='Test auxillary setting')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.data == 'MNIST':
        data_shape = (1, 28, 28)
    elif args.data == 'CelebA':
        data_shape = (3, 218, 178)
    else:
        raise NotImplementedError('Data shape unknown')
    
    if args.model == 'test':
        model = TestNetMNIST()
    elif args.model in ['AE', 'VAE']:
        """
        encoder = Encoder(encode_factor=3, base_channels=4, conv_blocks_per_decrease=2, channel_increase_factor=4)
        decoder = Decoder(final_size=(28,28), in_channels=256, encode_factor=3, channel_increase_factor=4, conv_blocks_per_decrease=2, RGB=False)
        model = nn.Sequential(encoder, decoder)
        """
        base_channels = 4
        conv_blocks_per_decrease = 6
        channel_increase_factor = 4
        encode_factor = 3
        encoding_dimension = 128
        initial_upsample_size = 4
        skip_connections = True
        model = Autoencoder(variational=args.model=='VAE', final_size=data_shape[1:3], base_channels=base_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, encoding_dimension=encoding_dimension, RGB=args.data!='MNIST', initial_upsample_size=initial_upsample_size, skip_connections=skip_connections)
        
    model.to(device)
    summary(model, data_shape, batch_size=args.batch_size)
    

"""
class TestNetMNIST(nn.Module):
    def __init__(self):
        super(TestNetMNIST, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.encoding = nn.Linear(512, 128)
        self.decoding = nn.Linear(128, 512)
        self.decode = nn.Sequential(
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 256, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample((7,7)),
            nn.ConvTranspose2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.ConvTranspose2d(128,64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 3),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0), 512)
        x = self.encoding(x)
        x = self.decoding(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.decode(x)
        return x
"""