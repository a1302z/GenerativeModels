import argparse as arg
import torch 
import torchvision
import models.Autoencoder as AE
import common.trainer as trainer
import torchsummary
import os, sys


parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'AE_linear', 'VAE', 'GAN'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', 'CelebA'), help='Data name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
parser.add_argument('--overfit', type=int, default=-1, help='Overfit to number of samples')
parser.add_argument('--resume_optimization', type=str, default=None, help='If the optimization of a already trained model should be continued give the model path')
parser.add_argument('--estimate_sizes', action='store_true', help='Print estimated size of model and data point.')
parser.add_argument('--device', type=str, default=None, help='Specify device')
args = parser.parse_args()

if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

#print(args)
## Parse config
config = {}
with open(args.config, 'r') as config_file:
    lines = config_file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        ls = line.split(' ')
        config[ls[0]] = ls[1]
print('Parsed config:\n  %s'%str(config))



##Create datasetloader
loader = None
overfit = args.overfit>-1
if overfit:
    config['batch_size']=1
if args.data == 'MNIST':
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=int(config['batch_size']), shuffle=not overfit)
elif args.data == 'CelebA':
    data_path = 'data/CelebA/'
    celeba = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(
        celeba,
        batch_size=int(config['batch_size']),
        shuffle=not overfit
    )
else: 
    raise NotImplementedError('The dataset you specified is not implemented')
print('Given %d training points (batch size: %d)'%(len(loader), int(config['batch_size'])))

RGB = not args.data == 'MNIST'
input_size = None
hidden_dim_size = None
encode_factor = None
base_channels = None
if args.data == 'MNIST':
    input_size = (28,28)
    hidden_dim_size = (64, 2)
    encode_factor = 2
    base_channels = 16
elif args.data == 'CelebA':
    input_size = (218, 178)
    hidden_dim_size = (64, 8)
    encode_factor = 4
    base_channels = 32
else:
    raise NotImplementedError('Input size/ Hidden dimension for dataset not specified')

## Init model
model = None
if args.model == 'AE':
    model = AE.Autoencoder(input_size, encode_factor, RGB = RGB)
elif args.model == 'AE_linear':
    model = AE.LinearAutoencoder(input_size, encode_factor, RGB = RGB, hidden_size=hidden_dim_size)
elif args.model == 'VAE':
    model = AE.VariationalAutoencoder(input_size, encode_factor, RGB = RGB, hidden_size=hidden_dim_size)
else:
    raise NotImplementedError('The model you specified is not implemented yet')
    
if args.estimate_sizes:
    #os.environ['CUDA_VISIBLE_DEVICES'] = 'CPU'
    sample = None
    for i, (data, target) in enumerate(loader):
        sample = data
        print('Data size (Byte): %s'%str(sys.getsizeof(data)))
        break
    """
    size_estimation = SizeEstimator(model, input_size=sample.size())
    print(size_estimation.estimate_size())
    """
    if torch.cuda.is_available():
        model.cuda()
    print(torchsummary.summary(model,(3 if RGB else 1, input_size[0], input_size[1])))


## Loss function
loss = None
if config['loss'] == 'L1':
    loss = torch.nn.functional.l1_loss
elif config['loss'] in ['MSE', 'L2']:
    loss = torch.nn.functional.mse_loss
elif config['loss'] == 'BCE':
    loss = torch.nn.BCELoss()
else:
    raise NotImplementedError('Loss not supported')
    
    
optim = trainer.train(args, loader, model, loss, config, num_overfit=args.overfit, resume_optim=args.resume_optimization, 
                      input_size=(3 if RGB else 1, input_size[0], input_size[1]))

