import argparse as arg
import torch 
import torchvision
import models.Autoencoder as AE
import common.trainer as trainer



parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'VAE'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', ''), help='Data name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
args = parser.parse_args()

#print(args)
## Parse config
config = {}
with open(args.config, 'r') as config_file:
    lines = config_file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        ls = line.split(' ')
        config[ls[0]] = ls[-1]
        if 'optimizer:' in ls[0]:
            optimizer = str(ls[-1])
        elif 'lr:' in ls[0]:
            lr = float(ls[-1])
        elif 'epochs:' in ls[0]:
            epochs = int(ls[-1])
        else: 
            print('Did not process: %s'%line)
print('Parsed config:\n  %s'%str(config))

##Create datasetloader
loader = None
if args.data == 'MNIST':
    loader = torchvision.datasets.MNIST(root = 'data', train = True, download = True)
else: 
    raise NotImplementedError('The dataset you specified is not implemented')
print('Given %d training points'%len(loader))


## Init model
model = None
if args.model == 'AE':
    model = AE.Autoencoder()
else:
    raise NotImplementedError('The model you specified is not implemented yet')
    
## Loss function
loss = None
if args.model == 'AE':
    loss = torch.nn.functional.l1_loss
    
    
#trainer.train(loader, model, loss, optimizer, epochs)