import torch
import torch.nn as nn
from models.common import init

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(ConvBlock, self).__init__()
        out = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
        if bn:
            out.append(nn.BatchNorm2d(out_channels))
        out.append(nn.LeakyReLU())
        self.enc = nn.Sequential(*out)
        
    def forward(self, x):
        return self.enc(x)
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=False):
        super(ConvTransposeBlock, self).__init__()
        out = [nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1)]
        if bn:
            out.append(nn.BatchNorm2d(out_channels))
        out.append(nn.LeakyReLU())
        self.dec = nn.Sequential(*out)
        
    def forward(self, x):
        return self.dec(x)

class Encoder(nn.Module):
    def __init__(self, base_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, RGB=False):
        super(Encoder, self).__init__()
        encode_modules = [ConvBlock(3 if RGB else 1, base_channels, bn = False)]
        cur_channels = base_channels
        for i in range(encode_factor):
            for j in range(conv_blocks_per_decrease):
                encode_modules.append(ConvBlock(cur_channels, cur_channels, True))
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
    def __init__(self, final_size=(28,28), in_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, RGB=False):
        super(Decoder, self).__init__()
        decode_modules = [nn.Upsample(scale_factor=2)]
        cur_channels = in_channels
        for i in range(encode_factor):
            for j in range(conv_blocks_per_decrease):
                decode_modules.append(ConvTransposeBlock(cur_channels, cur_channels, bn = True))
            decode_modules.append(ConvTransposeBlock(cur_channels, cur_channels//channel_increase_factor, bn=False))
            cur_channels = cur_channels//channel_increase_factor
            decode_modules.append(nn.Upsample(scale_factor=2))
        out_channels = 3 if RGB else 1
        decode_modules.append(ConvTransposeBlock(cur_channels, cur_channels, bn = False))
        decode_modules.append(nn.Conv2d(cur_channels, out_channels, 3, padding=1))
        decode_modules.append(nn.Sigmoid())
        decode_modules.append(nn.Upsample(final_size, mode='bilinear'))
        
        
        self.dec = nn.Sequential(*decode_modules)
        
    
    def forward(self, x):
        return self.dec(x)
    
    
class Autoencoder(nn.Module):
    def __init__(self, final_size=(28,28), base_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, encoding_dimension=128, RGB=False):
        super(Autoencoder, self).__init__()
        self.encoding_dimension = encoding_dimension
        self.encoder = Encoder(base_channels=base_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, RGB=RGB)
        self.final_channels = base_channels * (channel_increase_factor**encode_factor)
        self.decoder = Decoder(final_size=final_size, in_channels=self.final_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, RGB=RGB)
        self.to_hidden = nn.Linear(self.final_channels, encoding_dimension)
        self.from_hidden = nn.Linear(encoding_dimension, self.final_channels)
        
    def forward(self, x):
        x = self.encoder(x)
        s = x.size()
        x = x.view(x.size(0), -1)
        x = self.to_hidden(x)
        x = self.from_hidden(x)
        x = x.view(s)
        x = self.decoder(x)
        return x




    
if __name__ == '__main__':
    from torchsummary import summary
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, choices=['AE', 'AE_linear', 'VAE', 'test'], help='Which model to test?')
    #parser.add_argument('--input_dim', type=int, default=100, help='What input dimension for generator?')
    #parser.add_argument('--aux', action='store_true', help='Test auxillary setting')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'test':
        model = TestNetMNIST()
    elif args.model == 'AE':
        """
        encoder = Encoder(encode_factor=3, base_channels=4, conv_blocks_per_decrease=2, channel_increase_factor=4)
        decoder = Decoder(final_size=(28,28), in_channels=256, encode_factor=3, channel_increase_factor=4, conv_blocks_per_decrease=2, RGB=False)
        model = nn.Sequential(encoder, decoder)
        """
        base_channels = 4
        conv_blocks_per_decrease = 2
        channel_increase_factor = 4
        encode_factor = 3
        encoding_dimension = 128
        model = Autoencoder(final_size=(28,28), base_channels=base_channels, encode_factor=encode_factor, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, encoding_dimension=encoding_dimension, RGB=False)
        
    model.to(device)
    summary(model, (1, 28, 28))