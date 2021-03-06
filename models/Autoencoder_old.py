import torch
import torch.nn as nn
from models.common import init




"""
Module wrapper for nn.functional.interpolate
"""
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

"""
Halve input width and height
"""
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),       #a
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),                     #b
        )
        
    def forward(self, x):
        return self.down(x)

"""
Double input width and height
"""
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2),       #b
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, 3),           #a
        )
        
    def forward(self, x):
        return self.up(x)

"""
Shared implementation of convolutional encoder part
"""
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, encode_factor, base_channels=16, print_init=False):
        super(ConvEncoder, self).__init__()
        assert encode_factor > 0, 'Encode factor must be > 0'
        
        
        self.encoder = nn.ModuleList([DownBlock(in_channels, base_channels)])
        if print_init:
            print('Init Encoder')
            print('\tDown %d -> %d'%(in_channels, base_channels))
        for i in range(1, encode_factor):
            self.encoder.append(DownBlock(base_channels, base_channels*2))
            if print_init:
                print('\tDown %d -> %d'%(base_channels, base_channels*2))
            base_channels *= 2
        if print_init:
            print('\tFinish %d -> %d\n\n'%(base_channels, base_channels))
        
        self.finish_encoding = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 2),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        x = self.finish_encoding(x)
        return x
    
"""
Shared implementation of convolutional decoder part
"""
class ConvDecoder(nn.Module):
    def __init__(self, input_size, encode_factor, out_channels, base_channels=16, print_init=False):
        super(ConvDecoder, self).__init__()
        self.input_size = input_size
        assert encode_factor > 0, 'Encode factor must be > 0'
        
        
        base_channels = base_channels * (2**(encode_factor-1))
        
        if print_init:
            print('Init Decoder')
            print('\tStart %d->%d'%(base_channels,base_channels))
        self.start_decode = nn.Sequential(
            nn.ConvTranspose2d(base_channels,base_channels, 2),                #g
            nn.LeakyReLU(),
        )
        
        self.decoder = nn.ModuleList([])
        for i in range(1, encode_factor):
            self.decoder.append(UpBlock(base_channels, int(base_channels/2)))
            if print_init:
                print('\tUp %d -> %d'%(base_channels, base_channels/2))
            base_channels = int(base_channels/2)
        if print_init:
            print('\tUp %d -> %d'%(base_channels, out_channels))    
        self.decoder.append(UpBlock(base_channels, out_channels))

        self.finish = nn.Sequential(
            Interpolate(size=input_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.start_decode(x)
        for d in self.decoder:
            x = d(x)
        return self.finish(x)

"""
Fully Convolutional autoencoder
 -> Combination of ConvEncoder and ConvDecoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_size, encode_factor, RGB = True, base_channels=16, print_init=False):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = ConvEncoder(channels, encode_factor, base_channels=base_channels, print_init=print_init)
        
        self.decoder = ConvDecoder(input_size, encode_factor, channels, base_channels=base_channels, print_init=print_init)
        
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
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    
"""
Autoencoder with linear layer as encoding to reduce to hidden vector
"""
class LinearAutoencoder(nn.Module):
    # input size being tupel of image dimension
    def __init__(self, input_size, encode_factor, RGB = False, hidden_size=(64, 16), base_channels=16, print_init=False):
        super(LinearAutoencoder, self).__init__()
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = ConvEncoder(channels, encode_factor, base_channels=base_channels, print_init=print_init)
        
        self.h_hidden = input_size[0]
        self.w_hidden = input_size[1]
        for i in range(encode_factor):
            self.h_hidden = int(self.h_hidden/2)
            self.w_hidden = int(self.w_hidden/2)
        self.h_hidden -= 3
        self.w_hidden -= 3
        #width*height*channels
        hidden_channels = base_channels * (2**(encode_factor-1))
        if print_init:
            print('Hidden channels: %d'%hidden_channels)
            print('Hidden size: (%dx%d)'%(self.h_hidden, self.w_hidden))
        self.num_linear_input = self.h_hidden*self.w_hidden*hidden_channels
        
        self.tohidden = nn.Sequential(
            nn.Linear(self.num_linear_input, hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
        )
        self.fromhidden = nn.Sequential(
            nn.Linear(hidden_size[1], self.num_linear_input),
            nn.LeakyReLU()
        )
        self.decoder = ConvDecoder(input_size, encode_factor, channels, base_channels=base_channels, print_init=print_init)
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        self.s = x.size()
        #print(self.s)
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

    
"""
Implementation of Variational Autoencoder. 
Encoder predicts mean and variance
Decoder reconstructs input by sample from Gaussian normal distribution.
"""
class VariationalAutoencoder(nn.Module):
    # input size being tupel of image dimension
    def __init__(self, input_size, encode_factor, hidden_size=(64,16), RGB = False, base_channels=16, print_init=False):
        super(VariationalAutoencoder, self).__init__()
        self.hidden_size = hidden_size
        if RGB:
            channels = 3
        else: 
            channels = 1
        self.encoder = ConvEncoder(channels, encode_factor, base_channels=base_channels, print_init=print_init)
        
        
        self.h_hidden = input_size[0]
        self.w_hidden = input_size[1]
        for i in range(encode_factor):
            self.h_hidden = int(self.h_hidden/2)
            self.w_hidden = int(self.w_hidden/2)
        self.h_hidden -= 3 if encode_factor > 1 else 2
        self.w_hidden -= 3 if encode_factor > 1 else 2
        #width*height*channels
        hidden_channels = base_channels * (2**(encode_factor-1))
        if print_init:
            print('Hidden channels: %d'%hidden_channels)
            print('Hidden size: (%dx%d)'%(self.h_hidden, self.w_hidden))
        self.num_linear_input = self.h_hidden*self.w_hidden*hidden_channels
        
        
        self.hidden = nn.Linear(self.num_linear_input, hidden_size[0])
        self.mean = nn.Linear(hidden_size[0], hidden_size[1])
        self.log_var = nn.Linear(hidden_size[0], hidden_size[1])
        
        self.fromhidden = nn.Sequential(
            nn.Linear(hidden_size[1], hidden_size[0]),
            nn.Linear(hidden_size[0], self.num_linear_input)
        )
        
        self.decoder = ConvDecoder(input_size, encode_factor, channels, base_channels=base_channels, print_init=print_init)
        
    def forward(self, x):
        if len(x.size()) < 4:
            x = x.unsqueeze(0)
        #print(x.size())
        x = self.encoder(x)
        self.s = x.size()
        #print(self.s)
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
    
    
    
class EncoderNetMNIST(nn.Module):
    def __init__(self):
        super(EncoderNetMNIST, self).__init__()
        
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
        model = EncoderNetMNIST().to(device)
    elif args.model == 'AE':
        model = Autoencoder((28,28), 2, RGB = False, base_channels=128, print_init=False).to(device)
    elif args.model == 'VAE':
        model = VariationalAutoencoder((28, 28), 2, hidden_size=(128,64), RGB = False, base_channels=64, print_init=False).to(device)
    summary(model, (1, 28, 28))
        

        