
import torch 
import torchvision
import models.Autoencoder as AE
import models.GAN as GAN


def parse_config(path):
    config = {}
    with open(path, 'r') as config_file:
        lines = config_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            ls = line.split(' ')
            config[ls[0]] = ls[1]
    print('Parsed config:\n  %s'%str(config))
    return config


def create_model(config, model_name):
    RGB = config['RGB'] == 'True' if 'RGB' in config else None
    input_size = tuple(map(int, config['input_size'].split('x'))) if 'input_size' in config else None
    hidden_dim_size = tuple(map(int, config['hidden_dim_size'].split(','))) if 'hidden_dim_size' in config else None
    encode_factor = int(config['encode_factor']) if 'encode_factor' in config else None
    base_channels = int(config['base_channels']) if 'base_channels' in config else None
    """
    if args.data == 'MNIST':
        input_size = (28,28)
        hidden_dim_size = (64, 2)
        encode_factor = 2
        base_channels = 32
    elif args.data == 'CelebA':
        input_size = (218, 178)
        hidden_dim_size = (64, 8)
        encode_factor = 4
        base_channels = 32
    else:
        raise NotImplementedError('Input size/ Hidden dimension for dataset not specified')
    """

    ## Init model
    model = None
    if model_name == 'AE':
        model = AE.Autoencoder(input_size, encode_factor, RGB = RGB, base_channels=base_channels)
    elif model_name == 'AE_linear':
        model = AE.LinearAutoencoder(input_size, encode_factor, RGB = RGB, hidden_size=hidden_dim_size, base_channels=base_channels)
    elif model_name == 'VAE':
        model = AE.VariationalAutoencoder(input_size, encode_factor, RGB = RGB, hidden_size=hidden_dim_size, base_channels=base_channels)
    elif model_name == 'VanillaGAN':
        latent_dim = int(config.get('latent_dim', 10))
        #print('Latent dim was set to {:d}'.format(latent_dim))
        gen = GAN.VanillaGenerator(input_dim = latent_dim)
        disc = GAN.VanillaDiscriminator()
        model = (gen, disc)
    else:
        raise NotImplementedError('The model you specified is not implemented yet')
    return model


def create_dataset_loader(config, data, overfit=-1, ganmode=False):
    ##Create datasetloader
    loader = None
    overfit = overfit>-1
    #if overfit:
    #    config['batch_size']=1
    if data == 'MNIST':
        tfs = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
        if ganmode:
            print('Data normalized between -1 and 1')
            tfs.append(torchvision.transforms.Lambda(lambda x: x*0.616200 -0.738600))
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('data', train=True, download=True,
                           transform=torchvision.transforms.Compose(tfs)),
                            batch_size=int(config['batch_size']), shuffle=not overfit)
    elif data == 'CelebA':
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
    return loader

def create_test_loader(data='MNIST', directory='data'):
    if data == 'MNIST':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(directory, train=False, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
                            batch_size=1, shuffle=False)
    elif data == 'CelebA':
        data_path = 'data/CelebA/'
        celeba = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(
            celeba,
            batch_size=1,
            shuffle=False
        )
    else:
        raise NotImplementedError('Test dataset for specified dataset not implemented yet')
    
    return loader


def create_loss(config):
    loss = None
    if config['loss'] == 'L1':
        loss = torch.nn.functional.l1_loss
    elif config['loss'] in ['MSE', 'L2']:
        loss = torch.nn.functional.mse_loss
    elif config['loss'] == 'BCE':
        loss = torch.nn.BCELoss(reduction='mean')
    elif config['loss'] == 'CrossEntropy':
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError('Loss not supported')
    return loss