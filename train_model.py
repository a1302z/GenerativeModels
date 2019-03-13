import argparse as arg
import torch 
import torchvision
import models.Autoencoder as AE
import common.trainer as trainer



parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'AE_linear', 'VAE'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', ''), help='Data name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
parser.add_argument('--batch_size', type=int, help='specify batch size', default=1000)
parser.add_argument('--overfit', type=int, default=-1, help='Overfit to number of samples')
parser.add_argument('--resume_optimization', type=str, default=None, help='If the optimization of a already trained model should be continued give the model path')
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
overfit = args.overfit>-1
if overfit:
    args.batch_size=1
if args.data == 'MNIST':
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                        batch_size=args.batch_size, shuffle=not overfit)
else: 
    raise NotImplementedError('The dataset you specified is not implemented')
print('Given %d training points (batch size: %d)'%(len(loader), args.batch_size))


## Init model
model = None
if args.model == 'AE':
    model = AE.Autoencoder()
elif args.model == 'AE_linear':
    if args.data == 'MNIST':
        model = AE.LinearAutoencoder(input_size=(28,28), hidden_size=(128,2))
    else:
        raise NotImplementedError('Only for MNIST yet')
elif args.model == 'VAE':
    if args.data == 'MNIST':
        model = AE.VariationalAutoencoder(input_size=(28,28), hidden_size=(128,2))
    else:
        raise NotImplementedError('Only for MNIST yet')
else:
    raise NotImplementedError('The model you specified is not implemented yet')
    
## Loss function
loss = None
if config['loss'] == 'L1':
    loss = torch.nn.functional.l1_loss
elif config['loss'] == 'MSE':
    loss = torch.nn.functional.mse_loss
elif config['loss'] == 'BCE':
    loss = torch.nn.BCELoss(reduction='sum')
else:
    raise NotImplementedError('Loss not supported')
    
    
optim = trainer.train(args, loader, model, loss, config, num_overfit=args.overfit, resume_optim=args.resume_optimization)

