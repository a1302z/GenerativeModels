from sklearn.manifold import TSNE
import torch
import torchvision
from torchvision.utils import make_grid, save_image
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time
from configparser import ConfigParser
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--data', type=str, default='MNIST', choices=('MNIST', 'CelebA'), help='Data name to be used')
parser.add_argument('--model', type=str, choices=('AE', 'VAE'), required = True, help='Specify model')
parser.add_argument('--limit_datapoints', type=int, default=-1, help='limit number of data points')
args = parser.parse_args()

config_path = os.path.join(os.path.dirname(args.model_path), 'settings.ini')
config = ConfigParser()
config.read(config_path)
loader = setup.create_dataset_loader(config, data=args.data, test=True)
n_classes = 10 if config.getboolean('HYPERPARAMS', 'auxillary', fallback=False) and args.data == 'MNIST' else -1
model = setup.create_model(config, args.model, n_classes)
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

hidden_list = []
targets = []
for i, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
    #if i > 50:
    #    break
    #if i % int(len(loader)/10) == 0:
    #    print('%d/%d'%(i,len(loader)))
    data = data.to(device)
    if args.model == 'AE':
        hidden, c = model.encode(data)
    elif args.model == 'VAE':
        hidden, std, c = model.encode(data)
    if args.model == 'VAE':
        hidden = hidden[0]
    hidden = hidden.detach().cpu().numpy()
    hidden_list.append(hidden)
    targets.append(target[0])
    if args.limit_datapoints >= 0 and i >= args.limit_datapoints:
        break

hidden_list = np.vstack(hidden_list)
print('Computing tsne representation')
t = time.time()
X_embedded = TSNE(n_components=2).fit_transform(hidden_list)
t = time.time() -t
print('Took {:.1f} seconds'.format(t))

targets = np.stack(targets)
#plt.scatter(X_embedded[:,0], X_embedded[:,1], c=targets, alpha=0.3)

cm = matplotlib.cm.get_cmap('gist_rainbow')

fig, ax = plt.subplots()
for g in np.unique(targets):
    ix = np.where(targets == g)
    ax.scatter(X_embedded[ix,0], X_embedded[ix,1], color = cm(g/10.0), label = g, alpha=0.5)
ax.legend()

plt.savefig('result_figures/tsne_'+args.model+'_'+args.data+'.png')
plt.show()

