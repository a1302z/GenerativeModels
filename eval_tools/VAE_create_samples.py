"""
Create artificial output by sampling from standard normal distribution 
and using decoder of VAE
"""

import torch
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import copy
import os,sys,inspect
from configparser import ConfigParser
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained VAE')
parser.add_argument('--image_root', type=int, default=10, help='Number of images per axis (Squareroot of final #images)')
args = parser.parse_args()





model_name = 'VAE'
config_path = os.path.join(os.path.dirname(args.model_path), 'settings.ini')
config = ConfigParser()
config.read(config_path)
data = None
if config.get('TRAINING', 'input_size') == '28x28':
    data = 'MNIST'
elif config.get('TRAINING', 'input_size') == '218x178':
    data = 'CelebA'
else:
    raise NotImplementedError('Data size unknown')
n_classes = 10 if config.getboolean('HYPERPARAMS', 'auxillary', fallback=False) and data == 'MNIST' else -1
model = setup.create_model(config, model_name, n_classes)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

input_size = tuple(map(int, config.get('TRAINING', 'input_size').split('x')))
RGB = config.getboolean('TRAINING', 'RGB')
    

    
hidden_size = config.getint('HYPERPARAMS', 'latent_dim')
square = args.image_root
imgs = []
eps = torch.randn((square**2, hidden_size), device=device)
imgs = model.decode(eps).detach().cpu()
imgs = torch.clamp(imgs, 0, 1)
imgs = make_grid(imgs, nrow=square)
save_image(imgs, 'result_figures/artificial_samples_'+data+'.png')
plt.figure(figsize=(square, square))
cm = plt.cm.gray if data == 'MNIST' else None
plt.imshow(np.moveaxis(imgs.numpy(), 0, -1), cmap=cm)
plt.axis('off')
plt.show()
