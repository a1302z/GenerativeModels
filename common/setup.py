import os
import torch 
import torchvision
import models.Autoencoder as AE
import models.GAN as GAN


"""def parse_config(path):
    config = {}
    with open(path, 'r') as config_file:
        lines = config_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            ls = line.split(' ')
            config[ls[0]] = ls[1]
    print('Parsed config:\n  %s'%str(config))
    return config
"""

def create_model(config, model_name, num_classes=1):
    RGB = config.getboolean('TRAINING', 'RGB', fallback=None)
    input_size = tuple(map(int, config.get('TRAINING', 'input_size', fallback=None).split('x'))) 
    encode_blocks = config.getint('HYPERPARAMS', 'encode_blocks', fallback=None)
    base_channels = config.getint('HYPERPARAMS', 'base_channels', fallback=None)
    latent_dim = config.getint('HYPERPARAMS', 'latent_dim', fallback=10)
    channel_increase_factor = config.getint('HYPERPARAMS', 'channel_increase_factor', fallback=2)
    conv_blocks_per_decrease = config.getint('HYPERPARAMS', 'conv_blocks_per_decrease', fallback=1)
    initial_upsample_size = config.getint('HYPERPARAMS', 'initial_upsample_size', fallback=3)
    skip_connections = config.getboolean('HYPERPARAMS', 'skip_connections', fallback=False)
    num_classes = num_classes if config.getboolean('HYPERPARAMS', 'auxillary', fallback=False) else 0
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
    if model_name in ['AE', 'VAE']:
        model = AE.Autoencoder(variational = model_name == 'VAE', final_size=input_size, encode_factor=encode_blocks, RGB = RGB, base_channels=base_channels, channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, encoding_dimension=latent_dim, initial_upsample_size=initial_upsample_size, skip_connections=skip_connections, n_classes=num_classes)
    elif model_name == 'VanillaGAN':
        latent_dim = config.getint('HYPERPARAMS', 'latent_dim', fallback=10)
        #print('Latent dim was set to {:d}'.format(latent_dim))
        gen = GAN.VanillaGenerator(input_dim = latent_dim, num_classes=num_classes, blocks=encode_blocks, final_resolution=input_size,
                                   channel_increase_factor=channel_increase_factor, conv_blocks_per_decrease=conv_blocks_per_decrease, 
                                   initial_upsample_size=initial_upsample_size, skip_connections=skip_connections)
        disc = GAN.VanillaDiscriminator(n_classes=num_classes, blocks=config.getint('DISC_PARAMS', 'encode_blocks', fallback=encode_blocks), 
                                        channels=base_channels, up_blocks=encode_blocks,channel_increase_factor=channel_increase_factor,
                                        conv_blocks_per_decrease=config.getint('DISC_PARAMS', 'encode_blocks', fallback=conv_blocks_per_decrease), skip_connections=skip_connections, RGB=RGB)
        model = (gen, disc)
    elif model_name == 'DCGAN':
        gen = GAN.DCGenerator(input_dim = latent_dim, num_classes=num_classes)
        disc = GAN.DCDiscriminator(n_classes=num_classes)
        model = (gen, disc)
    else:
        raise NotImplementedError('The model you specified is not implemented yet')
    return model


def create_dataset_loader(config, data, overfit=-1, ganmode=False, test=False, directory='data'):
    ##Create datasetloader
    loader = None
    if overfit > 0:
        config['HYPERPARAMS']['batch_size']=str(overfit)
    if test:
        
        config['HYPERPARAMS']['batch_size']= str(1)
    if data == 'MNIST':
        tfs = [
            torchvision.transforms.ToTensor(),
        ]
        if ganmode:
            print('Data normalized between -1 and 1')
            tfs.append(torchvision.transforms.Normalize((0.5, ), (0.5, )))
            #tfs.append(torchvision.transforms.Lambda(lambda x: x*0.616200 -0.738600))
        else:
            tfs.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(directory, train=not test, download=True,
                           transform=torchvision.transforms.Compose(tfs)),
            batch_size=config.getint('HYPERPARAMS','batch_size') if not test else 1, shuffle=overfit<0 and not test,
            num_workers = 4  if not test else 0
        )
    elif data == 'CelebA':
        data_path = os.path.join(directory, 'CelebA/')
        tfs = [torchvision.transforms.ToTensor()]
        if ganmode:
            tfs.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            tfs.append(torchvision.transforms.Normalize((0.5060818,  0.42541984, 0.38281444), (0.31069, 0.29028416, 0.28965387)))
        celeba = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(
            celeba,
            batch_size=config.getint('HYPERPARAMS','batch_size') if not test else 1,
            shuffle=overfit<0 and not test, 
            num_workers=4 if not test else 0
        )
    else: 
        raise NotImplementedError('The dataset you specified is not implemented')
    print('Given %d training points (batch size: %d)'%(len(loader), config.getint('HYPERPARAMS', 'batch_size')))
    return loader

"""def create_test_loader(data='MNIST', directory='data'):
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
"""

def create_loss(config):
    loss = None
    if config.get('HYPERPARAMS', 'loss') == 'L1':
        loss = torch.nn.functional.l1_loss
    elif config.get('HYPERPARAMS', 'loss') in ['MSE', 'L2']:
        loss = torch.nn.functional.mse_loss
    elif config.get('HYPERPARAMS', 'loss') == 'BCE':
        loss = torch.nn.BCELoss()
    elif config.get('HYPERPARAMS', 'loss') == 'CrossEntropy':
        loss = torch.nn.CrossEntropyLoss()
    elif config.get('HYPERPARAMS', 'loss') == 'NLL':
        loss = torch.nn.NLLLoss()
    else:
        raise NotImplementedError('Loss not supported')
    return loss