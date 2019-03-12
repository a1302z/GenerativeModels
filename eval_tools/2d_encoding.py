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
import models.Autoencoder as AE



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--model', type=str, choices=('AE_linear', 'VAE'), required = True, help='Specify model')
args = parser.parse_args()

loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=1, shuffle=False)
if args.model == 'AE':
    raise NotImplementedException('Not possible in this model')
    #model = AE.Autoencoder()
elif args.model == 'AE_linear':
    model = AE.LinearAutoencoder(input_size=(28,28), hidden_size=(128,2))
elif args.model == 'VAE':
    model = AE.VariationalAutoencoder(input_size=(28,28), hidden_size=(128,2))
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
