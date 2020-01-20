import torch
import torchvision
import numpy as np
import tqdm
import matplotlib.pyplot as plt

tfs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5,), (0.5,))
    torchvision.transforms.Normalize((0.130661,), (0.308108,)),
    torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
])

dset = torchvision.datasets.MNIST('data', train=True, download=True,
                           transform=tfs)


loader = torch.utils.data.DataLoader(
            dset,
            batch_size=1, shuffle=False)
acc = np.zeros((1, 28, 28))
sq_acc = np.zeros((1, 28, 28))
min_px, max_px = 1e10, -1e10
imgs = []
for data, _ in tqdm.tqdm(loader, total=len(loader)):
    imgs.append(data.numpy())
imgs = np.vstack(imgs)
print('stacked')
mean = np.mean(imgs)
std = np.std(imgs)
print('Data in range [{:.2f}, {:.2f}]'.format(np.min(imgs), np.max(imgs)))
print('Mean: {:.8f}\nstd: {:.8f}'.format(mean, std))
imgs = imgs.flatten()
plt.hist(imgs, bins=100)
plt.show()
"""
    acc += np.sum(imgs, axis=0)
    sq_acc += np.sum(imgs**2, axis=0)
    min_px = min(min_px, imgs.min())
    max_px = max(max_px, imgs.max())
    
print('Dataset range from {:.2f} to {:.2f}'.format(min_px, max_px))
    
N = len(dset) * acc.shape[1] * acc.shape[2]

mean_p = np.asarray([np.sum(acc[c]) for c in range(1)])
mean_p /= N
print('Mean pixel = {:.6f}'.format(float(mean_p)))

# std = E[x^2] - E[x]^2
std_p = np.asarray([np.sum(sq_acc[c]) for c in range(1)])
std_p /= N
std_p -= (mean_p ** 2)
print('Std. pixel = {:.6f}'.format(float(std_p)))
"""