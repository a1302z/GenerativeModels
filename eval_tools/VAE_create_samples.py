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
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained VAE')
parser.add_argument('--image_root', type=int, default=10, help='Number of images per axis (Squareroot of final #images)')
args = parser.parse_args()


model_name = 'VAE'
config_path = os.path.join(os.path.dirname(args.model_path), 'settings.config')
config = setup.parse_config(config_path)
model = setup.create_model(config, model_name)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

input_size = tuple(map(int, config['input_size'].split('x')))
RGB = config['RGB'] == 'True'
test_sample = torch.zeros([1, 3 if RGB else 1, input_size[0], input_size[1]])
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()
    test_sample = test_sample.cuda()
#VAE needs one encoding operation before decoding
model(test_sample)
    

    
hidden_size = tuple(map(int, config['hidden_dim_size'].split(',')))
square = args.image_root
imgs = []
for i in range(square**2):
    eps = torch.randn(hidden_size[1])
    if cuda:
        eps = eps.cuda()
    img = model.decode(eps)
    img = img.squeeze(0).detach().cpu()
    imgs.append(img)
imgs = make_grid(imgs, nrow=square)
data = None
if config['input_size'] == '28x28':
    data = 'MNIST'
elif config['input_size'] == '218x178':
    data = 'CelebA'
else:
    raise NotImplementedError('Data size unknown')
save_image(copy.deepcopy(imgs), 'result_figures/artificial_samples_'+data+'.png')
plt.figure(figsize=(square, square))
cm = plt.cm.gray if data == 'MNIST' else None
plt.imshow(np.moveaxis(imgs.numpy(), 0, -1), cmap=cm)
plt.axis('off')
plt.show()
