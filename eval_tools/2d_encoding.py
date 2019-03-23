import torch
import torchvision
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--model', type=str, choices=('AE_linear', 'VAE'), required = True, help='Specify model')
args = parser.parse_args()

config_path = os.path.join(os.path.dirname(args.model_path), 'settings.config')
config = setup.parse_config(config_path)
loader = setup.create_test_loader(data='MNIST')
assert tuple(map(int, config['hidden_dim_size'].split(',')))[1] == 2, 'Hidden dimension not 2D'
model = setup.create_model(config, args.model)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()


points = []
targets = []
for i, (data, target) in enumerate(loader):
    if i % 1000 == 0:
        print('%d/%d'%(i, len(loader)))
    if args.model == 'VAE':
        x, log_var = model.encode(data.cuda())
    else:
        x = model.encode(data.cuda())
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
