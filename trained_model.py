import argparse as arg
import torch 
import torchvision
import models.Autoencoder as AE
import common.trainer as trainer
import datetime



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
        config[ls[0]] = ls[1]
print('Parsed config:\n  %s'%str(config))

##Create datasetloader
loader = None
if args.data == 'MNIST':
    transform = torchvision.transforms.ToTensor()
    loader = torchvision.datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
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
    
    
optim = trainer.train(loader, model, loss, config)

##save model
timestamp = str(datetime.datetime.now()).replace(' ', '_')
save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
torch.save(save_dict, 'trained_models/'+args.model+'_'+args.data+'_'+timestamp)