import argparse as arg
import torch
import common.trainer as trainer
import common.setup as setup
import os, sys


parser = arg.ArgumentParser(description='Train generative model.')
parser.add_argument('--model', type=str, choices=('AE', 'AE_linear', 'VAE', 'GAN'), help='Which model to train', required=True)
parser.add_argument('--data', type=str, choices=('MNIST', 'CelebA'), help='Data name to be used for training', required=True)
parser.add_argument('--config', type=str, help='path to config file', required=True)
parser.add_argument('--overfit', type=int, default=-1, help='Overfit to number of samples')
parser.add_argument('--resume_optimization', type=str, default=None, help='If the optimization of a already trained model should be continued give the model path')
parser.add_argument('--device', type=str, default=None, help='Specify device')
args = parser.parse_args()

if args.device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

#print(args)
## Parse config
config = setup.parse_config(args.config)
RGB = config['RGB'] == 'True'
input_size = tuple(map(int, config['input_size'].split('x')))
##Create datasetloader
loader = setup.create_dataset_loader(config, args.data, args.overfit)
##Create model
model = setup.create_model(config, args.model)
## Loss function
loss = setup.create_loss(config)

    
optim = trainer.train(args, loader, model, loss, config, num_overfit=args.overfit, resume_optim=args.resume_optimization, 
                      input_size=(3 if RGB else 1, input_size[0], input_size[1]))

