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
parser.add_argument('--batch_size', type=int, help='specify batch size', default=20)
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
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=args.batch_size, shuffle=True)
else: 
    raise NotImplementedError('The dataset you specified is not implemented')
print('Given %d training points (batch size: %d)'%(len(loader), args.batch_size))


## Init model
model = None
if args.model == 'AE':
    model = AE.Autoencoder()
else:
    raise NotImplementedError('The model you specified is not implemented yet')
    
## Loss function
loss = None
if config['loss'] == 'L1':
    loss = torch.nn.functional.l1_loss
else:
    raise NotImplementedError('Loss not supported')
    
    
optim = trainer.train(loader, model, loss, config)

##save model
timestamp = str(datetime.datetime.now()).replace(' ', '_')
save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
path = 'trained_models/'+args.model+'_'+args.data+'_'+timestamp
print("Save model to path %s"%path)
torch.save(save_dict, path)