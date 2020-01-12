import torchvision
import tqdm

mnist = torchvision.datasets.MNIST('data', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                               torchvision.transforms.Lambda(lambda x: x*0.616200 -0.738600)
                               #torchvision.transforms.Lambda(lambda x: (x+0.424213)/3.245700)
                           ]))

mn, mx = 1e6, -1e6
for d, t in tqdm.tqdm(mnist, total=len(mnist)):
    n, x = d.min().item(), d.max().item()
    mn = min(mn, n)
    mx = max(mx, x)
print('MNIST data between {:.6f} and {:.6f}'.format(mn, mx))
print('Therefore multiply by {:.6f} and add {:.6f}'.format(2/(mx-mn), -(2*mn)/(mx-mn)-1))