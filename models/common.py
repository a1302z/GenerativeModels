import torch.nn as nn

def init(model):
    print('Initializing weights for {:s}'.format(str(model.__class__.__name__)))
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)