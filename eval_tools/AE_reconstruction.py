import torch
import torchvision
from torchvision.utils import make_grid, save_image
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup
import copy

def normalize(vector):
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalised = (vector - min_v) / range_v
    else:
        normalised = torch.zeros(vector.size())
    return normalised

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--model', type=str, choices=('AE','AE_linear', 'VAE'), required = True, help='Specify model')
parser.add_argument('--data', type=str, default='MNIST', choices=('MNIST', 'CelebA'), help='Data name to be used')
parser.add_argument('--num_pics', type=int, default=15, help='How many pics to show in evaluation?')
args = parser.parse_args()

config_path = os.path.join(os.path.dirname(args.model_path), 'settings.config')
config = setup.parse_config(config_path)
loader = setup.create_test_loader(data=args.data)
model = setup.create_model(config, args.model)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()

img_list = []

for i, (data, target) in enumerate(loader):
    print('Image %d/%d (Target: %d)'%(i,args.num_pics, target[0]))
    img_list.append(data.squeeze(0))
    if cuda:
        data = data.cuda()
    if args.model == 'VAE':
        mu, log_var = model.encode(data)
        z = model.sample_z(mu, log_var)
        reconstructed = model.decode(z)
    else:
        reconstructed = model(data)
     
    rec_img = reconstructed.squeeze(0).detach().cpu()
    #diff = torch.abs(reconstructed - data).squeeze(0).detach().cpu()
    img_list.append(rec_img)
    #img_list.append(diff)
    if i+1 >= args.num_pics:
        break

img = make_grid(img_list, nrow=2, scale_each=True)
save_image(copy.deepcopy(img), 'result_figures/'+args.model+'_'+args.data+'_reconstruction.png')
if args.data == 'MNIST':
    img = np.sum(img.numpy(), axis=0)
else:
    img = img.numpy()
    img = np.moveaxis(img, 0, -1)
ratio = float(config['input_size'][1])/float(config['input_size'][0])
plt.figure(figsize=(3.0*args.num_pics*ratio, args.num_pics))
cmap = plt.cm.gray if args.data == 'MNIST' else None
plt.imshow(img, cmap=cmap)
plt.axis('off')
plt.show()


