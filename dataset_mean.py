import torch
import torchvision
import numpy as np
import tqdm
import matplotlib.pyplot as plt


import argparse as arg
parser = arg.ArgumentParser()
parser.add_argument('--data', default='MNIST', choices=['MNIST', 'CelebA'], help='Which dataset')
parser.add_argument('--batch_size', default=128, type=int, help='Size of batches')
args = parser.parse_args()

if args.data == 'MNIST':
    tfs = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5,), (0.5,))
        #torchvision.transforms.Normalize((0.130661,), (0.308108,)),
        #torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])

    dset = torchvision.datasets.MNIST('data', train=True, download=True,
                               transform=tfs)
    channels = 1
elif args.data == 'CelebA':
    dset = torchvision.datasets.ImageFolder(
            root='data/CelebA/',
            transform=torchvision.transforms.ToTensor()
        )
    channels = 3
else:
    raise NotImplementedError('Dataset not implemented')

loader = torch.utils.data.DataLoader(
            dset,
            batch_size=args.batch_size, shuffle=False)

"""
Credits to https://forums.fast.ai/t/normalizing-your-dataset/49799
"""

cnt = 0
fst_moment = torch.empty(channels)
snd_moment = torch.empty(channels)
mn, mx = 1e10, -1e10
for images, _ in tqdm.tqdm(loader, total=len(loader)):
    b, c, h, w = images.size()
    nb_pixels = b * h * w
    sum_ = torch.sum(images, dim=[0, 2, 3])
    sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3] )
    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
    cnt += nb_pixels
    mn = min(mn, images.min().item())
    mx = max(mx, images.max().item())
std = torch.sqrt(snd_moment - fst_moment ** 2).numpy()
fst_moment = fst_moment.numpy()
print('Mean: {:s}\tStd: {:s}'.format(str(fst_moment), str(st)))
print('Data range in [{:.3f}, {:.3f}]'.format(mn, mx))


""" First version with accumulating data - only works for small dataset such as MNIST
imgs = []
for data, _ in tqdm.tqdm(loader, total=len(loader)):
    imgs.append(data.numpy())
imgs = np.vstack(imgs)
print('stacked')
mean = np.mean(imgs)
std = np.std(imgs)
print('Mean: {:.8f}\nstd: {:.8f}'.format(mean, std))
print('Data in range [{:.2f}, {:.2f}]'.format(np.min(imgs, axis=0), np.max(imgs, axis=0)))

imgs = imgs.flatten()
plt.hist(imgs, bins=100)
plt.show()
"""