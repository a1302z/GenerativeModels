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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='path to trained autoencoder')
parser.add_argument('--model', type=str, choices=('AE','AE_linear'), required = True, help='Specify model')
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
    model = AE.LinearAutoencoder(input_size=(28,28))
else:
    raise NotImplementedError('Model not supported')
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

img_list = []
for i, (data, target) in enumerate(loader):
    if i % int(len(loader)/args.num_pics) == 0:
        print('%d/%d'%(i/int(len(loader)/args.num_pics),args.num_pics))
        print('Target: %d'%target[0])
        reconstructed = model(data)
        #hidden = model.encode(data)
        #print(data.squeeze(0).size())
        #print(reconstructed.squeeze(0).size())
        rec_img = reconstructed.squeeze(0).detach()
        data_img = data.squeeze(0)

        mi, ma = data_img.min(), data_img.max()
        #print("(data) Min: %f\t Max: %f"%(mi,ma))
        mi, ma = rec_img.numpy().min(), rec_img.numpy().max()
        #rec_img = (rec_img-mi)/(ma-mi)
        #mi, ma = rec_img.numpy().min(), rec_img.numpy().max()
        #print("(recu) Min: %f\t Max: %f"%(mi,ma))
        img_list.append(torch.stack([data_img.cpu(), rec_img], 0))

img = make_grid(torch.cat(img_list, 0), nrow=2)
save_image(img, 'result_figures/'+args.model+'_reconstruction.png')
img = np.moveaxis(img.numpy(), 0, -1)
plt.imshow(img)
plt.show()


