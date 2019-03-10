from sklearn.manifold import TSNE
import torch
import torchvision
from torchvision.utils import make_grid, save_image
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import models.Autoencoder as AE



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
#parser.add_argument('--model', type=str, choices=('AE','AE_linear'), required = True, help='Specify model')
#parser.add_argument('--num_pics', type=int, default=50, help='How many pics to show in evaluation?')
args = parser.parse_args()

loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=1, shuffle=False)
model = AE.LinearAutoencoder(input_size=(28,28))
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
cuda = torch.cuda.is_available()
if cuda:
    model.cuda()

hidden_list = []
targets = []
for i, (data, target) in enumerate(loader):
    #if i > 50:
    #    break
    if i % int(len(loader)/10) == 0:
        print('%d/%d'%(i,len(loader)))
    if cuda:
        data = data.cuda()
        hidden = model.encode(data).cpu().detach().numpy()
    else:
        hidden = model.encode(data).detach().numpy()
    hidden_list.append(hidden)
    targets.append(target[0])

hidden_list = np.vstack(hidden_list)
print('Computing tsne representation')
X_embedded = TSNE(n_components=2).fit_transform(hidden_list)


targets = np.stack(targets)
#plt.scatter(X_embedded[:,0], X_embedded[:,1], c=targets, alpha=0.3)

cm = matplotlib.cm.get_cmap('gist_rainbow')

fig, ax = plt.subplots()
for g in np.unique(targets):
    ix = np.where(targets == g)
    ax.scatter(X_embedded[ix,0], X_embedded[ix,1], color = cm(g/10.0), label = g, alpha=0.5)
ax.legend()

plt.savefig('result_figures/tsne_autoencoder.png')
plt.show()

