import torch
import torchsummary
import argparse
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import common.setup as setup
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='path to config file', required=True)
parser.add_argument('--model', type=str, choices=('AE','AE_linear', 'VAE'), required = True, help='Specify model')
args = parser.parse_args()


config = setup.parse_config(args.config)
RGB = config['RGB'] == 'True'
input_size = tuple(map(int, config['input_size'].split('x')))

model = setup.create_model(config, args.model)

sample = None
"""
for i, (data, target) in enumerate(loader):
    sample = data
    print('Data size (Byte): %s'%str(sys.getsizeof(data)))
    break
"""
if torch.cuda.is_available():
    model.cuda()
print(torchsummary.summary(model,(3 if RGB else 1, input_size[0], input_size[1])))

