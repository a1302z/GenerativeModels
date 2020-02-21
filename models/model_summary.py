"""
Run by python -m models.model_summary from main folder
"""

import argparse as arg
from torch.cuda import is_available as cuda_available
from torchsummary import summary
from configparser import ConfigParser
from common.setup import create_model


parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'VAE', 'VanillaGAN', 'DCGAN'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', 'CelebA'), help='Dataset name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
args = parser.parse_args()


config = ConfigParser()
config.read(args.config)

RGB = config.getboolean('TRAINING', 'RGB')
input_size = tuple(map(int, config.get('TRAINING', 'input_size').split('x')))
batch_size = config.getint('HYPERPARAMS', 'batch_size')
device = 'cuda' if cuda_available() else 'cpu'
##Create model
n_classes = 1
if args.data == 'MNIST':
    n_classes = 10
model = create_model(config, args.model, num_classes=n_classes)
if args.model in ['VanillaGAN', 'DCGAN']:
    gen, disc = model
    summary(gen.to(device), (config.getint('HYPERPARAMS', 'latent_dim'),), batch_size=batch_size, device=device)
    summary(disc.to(device), (3 if RGB else 1,*input_size), device=device)
elif args.model in ['AE', 'VAE']:
    summary(model.to(device), (3 if RGB else 1,*input_size), batch_size=batch_size, device=device)
else:
    raise NotImplementedError('Model not supported by this script yet')
    
