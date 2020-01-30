import torch.nn as nn

def init(model):
    print('Initializing weights for {:s}'.format(str(model.__class__.__name__)))
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
                
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
        init(self)
        
    def forward(self, x):
        return self.enc(x)
    
class Decoder(nn.Module):    
    def __init__(self, final_size=(28,28), in_channels=16, encode_factor=2, channel_increase_factor=2, conv_blocks_per_decrease=1, RGB=False, initial_upsample_size=3, skip_connections=False, final_activation = nn.Sigmoid()):
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
        decode_modules.append(final_activation)
        
        
        self.dec = nn.Sequential(*decode_modules)
        init(self)
        
    
    def forward(self, x):
        return self.dec(x)