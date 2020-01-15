
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
    section_training = config['TRAINING']
    RGB = section_training.getboolean('RGB') if 'RGB' in section_training else None
    input_size = tuple(map(int, section_training.get('input_size').split('x'))) if 'input_size' in section_training else None
    section_hyper = config['HYPERPARAMS']
    hidden_dim_size = tuple(map(int, section_hyper.get('hidden_dim_size').split(','))) if 'hidden_dim_size' in section_hyper else None
    encode_factor = section_hyper.getint('encode_factor', None)
    base_channels = section_hyper.getint('base_channels', None)
    latent_dim = section_hyper.getint('latent_dim', 10)
    num_classes = num_classes if config.getboolean('GAN_HACKS', 'auxillary', fallback=False) else 0
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
        latent_dim = config.getint('HYPERPARAMS', 'latent_dim', fallback=10)
        #print('Latent dim was set to {:d}'.format(latent_dim))
        gen = GAN.VanillaGenerator(input_dim = latent_dim, num_classes=num_classes)
        disc = GAN.VanillaDiscriminator(n_classes=num_classes)
        model = (gen, disc)
    elif model_name == 'DCGAN':
        gen = GAN.DCGenerator(input_dim = latent_dim, num_classes=num_classes)
        disc = GAN.DCDiscriminator(n_classes=num_classes)
        model = (gen, disc)
    else:
        raise NotImplementedError('The model you specified is not implemented yet')
    return model


def create_dataset_loader(config, data, overfit=-1, ganmode=False):
    ##Create datasetloader
    loader = None
    if overfit > 0:
        config['HYPERPARAMS']['batch_size']=str(overfit)
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
            batch_size=config.getint('HYPERPARAMS','batch_size'), shuffle=overfit>-1,
            num_workers = 4
        )
    elif data == 'CelebA':
        data_path = 'data/CelebA/'
        celeba = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )
        loader = torch.utils.data.DataLoader(
            celeba,
            batch_size=int(config['batch_size']),
            shuffle=not overfit, 
            num_workers=4
        )
    else: 
        raise NotImplementedError('The dataset you specified is not implemented')
    print('Given %d training points (batch size: %d)'%(len(loader), config.getint('HYPERPARAMS', 'batch_size')))
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
    if config.get('HYPERPARAMS', 'loss') == 'L1':
        loss = torch.nn.functional.l1_loss
    elif config.get('HYPERPARAMS', 'loss') in ['MSE', 'L2']:
        loss = torch.nn.functional.mse_loss
    elif config.get('HYPERPARAMS', 'loss') == 'BCE':
        loss = torch.nn.BCELoss(reduction='mean')
    elif config.get('HYPERPARAMS', 'loss') == 'CrossEntropy':
        loss = torch.nn.CrossEntropyLoss()
    elif config.get('HYPERPARAMS', 'loss') == 'NLL':
        loss = torch.nn.NLLLoss()
    else:
        raise NotImplementedError('Loss not supported')
    return loss