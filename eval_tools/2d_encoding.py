import torch
import torchvision
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from configparser import ConfigParser
import tqdm
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--model', type=str, choices=('AE', 'VAE'), required = True, help='Specify model')
args = parser.parse_args()

config_path = os.path.join(os.path.dirname(args.model_path), 'settings.ini')
config = ConfigParser()
config.read(config_path)
assert config.getint('HYPERPARAMS', 'latent_dim') == 2, 'latent dim needs to be 2 for this script'
assert 'MNIST' in args.model_path, 'model needs to be trained on MNIST'
loader = setup.create_dataset_loader(config, data='MNIST', test=True)
n_classes = 10 if config.getboolean('HYPERPARAMS', 'auxillary', fallback=False) else -1
model = setup.create_model(config, args.model, n_classes)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


points = []
targets = []
for i, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
    data = data.to(device)
    if args.model == 'VAE':
        x, log_var, c = model.encode(data)
    else:
        x, c = model.encode(data)
    points.append(x.detach().cpu().numpy())
    targets.append(target)
    #if i > 10:
    #    break
points = np.vstack(points)
targets = np.stack(targets)

cm = matplotlib.cm.get_cmap('gist_rainbow')
fig, ax = plt.subplots()
for g in np.unique(targets):
    ix = np.where(targets == g)
    ax.scatter(points[ix,0], points[ix,1], color = cm(g/10.0), label = g, alpha=0.5)
ax.legend()
plt.savefig('result_figures/2d_encoding_'+args.model+'.png')
plt.show()
