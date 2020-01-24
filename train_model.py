import argparse as arg
import torch
import numpy as np
from configparser import ConfigParser
import common.trainer as trainer
import common.setup as setup
import os, sys


parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'VAE', 'VanillaGAN', 'DCGAN'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', 'CelebA'), help='Dataset name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
parser.add_argument('--overfit', type=int, default=-1, help='Overfit to number of samples')
parser.add_argument('--resume_optimization', type=str, default=None, help='If the optimization of a already trained model should be continued give the model path')
parser.add_argument('--device', type=str, default=None, help='Specify device')
args = parser.parse_args()

if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
"""
Make training deterministic
"""
seed = 1
#random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#print(args)
## Parse config
config = ConfigParser()
config.read(args.config)
RGB = config.getboolean('TRAINING', 'RGB')
input_size = tuple(map(int, config.get('TRAINING', 'input_size').split('x')))
##Create model
n_classes = 1
if args.data == 'MNIST':
    n_classes = 10
model = setup.create_model(config, args.model, num_classes=n_classes)
##Create datasetloader
loader = setup.create_dataset_loader(config, args.data, args.overfit,ganmode='gan' in args.model.lower())
## Loss function
loss = setup.create_loss(config)

if args.model in ['AE', 'VAE']:
    optim = trainer.train_AE(args, loader, model, loss, config, num_overfit=args.overfit, resume_optim=args.resume_optimization, 
                      input_size=(3 if RGB else 1, input_size[0], input_size[1]))
elif args.model in ['VanillaGAN', 'DCGAN']:
    optim = trainer.train_GAN(args, loader, model[0], model[1], loss, config, num_overfit=args.overfit, resume_optim=args.resume_optimization, 
                      input_size=(3 if RGB else 1, input_size[0], input_size[1]))
else:
    raise NotImplementedError('Model not implemented')
