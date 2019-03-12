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
import models.Autoencoder as AE

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
parser.add_argument('--num_pics', type=int, default=5, help='How many pics to show in evaluation?')
args = parser.parse_args()

loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=False, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=1, shuffle=False)
if args.model == 'AE':
    model = AE.Autoencoder()
elif args.model == 'AE_linear':
    model = AE.LinearAutoencoder(input_size=(28,28), hidden_size=(128,2))
elif args.model == 'VAE':
    model = AE.VariationalAutoencoder(input_size=(28,28), hidden_size=(128,2))
else:
    raise NotImplementedError('Model not supported')
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

cuda = torch.cuda.is_available()
if cuda:
    model.cuda()

img_list = []
for i, (data, target) in enumerate(loader):
    if i % int(len(loader)/args.num_pics) == 0:
        print('Image %d/%d (Target: %d)'%(i/int(len(loader)/args.num_pics),args.num_pics, target[0]))
        img_list.append(data.squeeze(0))
        if cuda:
            data = data.cuda()
        if args.model == 'VAE':
            mu, log_var = model.encode(data)
            z = model.sample_z(mu, log_var)
            reconstructed = model.decode(z)
        else:
            reconstructed = model(data)
        
        #hidden = model.encode(data)
        #print(data.squeeze(0).size())
        #print(reconstructed.squeeze(0).size())
        rec_img = reconstructed.squeeze(0).detach().cpu()
        #data_img = normalize(data.squeeze(0)).cpu()

        #mi, ma = data_img.min(), data_img.max()
        #print("(data) Min: %f\t Max: %f"%(mi,ma))
        #mi, ma = rec_img.numpy().min(), rec_img.numpy().max()
        #print("(recu) Min: %f\t Max: %f"%(mi,ma))
        img_list.append(rec_img)

img = make_grid(img_list, nrow=2)*255
save_image(img, 'result_figures/'+args.model+'_reconstruction.png')
img = np.sum(img.numpy(), axis=0)
plt.figure(figsize=(2.0, args.num_pics))
plt.imshow(img, cmap=plt.cm.gray)
plt.show()


