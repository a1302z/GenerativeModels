import torch
import models.Autoencoder as AE
import visdom
import matplotlib.pyplot as plt
import datetime, time
import numpy as np
import os, sys
import gc
import shutil
import tqdm
import warnings

def save_model(model, optim, save_dir, name):
    save_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}
    path = save_dir + '/'+name
    torch.save(save_dict, path)
    #print('Saved model %s to %s'%(name, path))


def train_AE(args, dataset, model, loss_fn, config, num_overfit=-1, resume_optim=None, input_size=(1, 28,28)):
    lr=config.getfloat('HYPERPARAMS', 'lr')
    print("Learning rate is %f"%lr)
    
    """
    Create optimizer
    """
    opt_list = list(model.parameters())
    if type(model) == AE.VariationalAutoencoder:
        #kl_loss_weight = torch.tensor(float(config['VAE_loss_weight']))
        weighted_loss = AE.WeightedMultiLoss(init_values=[config.getfloat('HYPERPARAMS', 'IMG_loss_weight'), config.getfloat('HYPERPARAMS', 'VAE_loss_weight')], learn_weights=config.getboolean('HYPERPARAMS', 'learn_loss_weights'))
        opt_list = list(opt_list +list(weighted_loss.parameters()))
    if config['optimizer'] == 'SGD':
        optim = torch.optim.SGD(params=opt_list, lr=lr)
    elif config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(params=opt_list, lr=lr)
    else:
        print('Unknown optimizer!')
    ##init model
    #AE.init(model)
    if resume_optim is not None:
        checkpoint = torch.load(resume_optim)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model/Optimizer loaded")
        
        
    """
    Initialize training parameters
    """
    overfit = num_overfit > -1
    save_interval = config.getint('TRAINING', 'save_epoch_interval', fallback=10)
    ##create save folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    path = args.model+'_'+args.data+'_'+timestamp
    if overfit:
        path += '_overfitted'
    save_dir = 'trained_models/'+path
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copyfile(args.config, save_dir+'/settings.config')
    except OSError:
        print ('Error: Creating directory. ' + save_dir)
    print("Created save directory at %s"%save_dir)
    
    cuda = torch.cuda.is_available()
    if cuda:
        print("CUDA available")
        model.cuda()
        if type(model) == AE.VariationalAutoencoder:
            #kl_loss_weight = kl_loss_weight.cuda()
            weighted_loss.cuda()
        if resume_optim is not None:
            for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                             state[k] = v.cuda()
        
    """
    Initialize visdom
    """
    use_vis = config.getboolean('TRAINING', 'visdom', fallback=True)
    if use_vis:
        print('Use visdom')
        vis_env = 'training/{:s}_{:s}_{:s}'.format(args.model, args.data, timestamp)
        vis = visdom.Visdom()
        assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
        plt_dict = dict(name='training loss',ytickmax=10, xlabel='epoch', ylabel='loss', legend=['train_loss'])
        vis_plot = vis.line(Y=[0],env=vis_env,opts=plt_dict)
        vis_image = vis.image(np.zeros(input_size),env=vis_env)
        if type(model) == AE.VariationalAutoencoder:
            vae_dict = dict(ytickmax=10, legend=['reconstruction loss', 'kl loss'], xlabel='epoch', ylabel='loss')
            vis_vae = vis.line(Y=[0,0], env=vis_env, opts=vae_dict)
            vae_w_dict = dict(ymin=0, ymax=10, legend=['Weight Image', 'Weight KL-div'], xlabel='epoch', ylabel='value')
            vis_weights = vis.line(Y=[0,0], env=vis_env, opts=vae_w_dict)
            
    log_every = max(1, int(len(dataset)/config.getint('TRAINING', 'log_every_dataset_chunk', fallback=5))) if not overfit else num_overfit
    if 'epochs' not in config['HYPERPARAMS']:
        warnings.warn('Number of epochs not specified in config - Use 20 as default', Warning)
    epochs = config.getint('HYPERPARAMS', 'epochs', fallback=20)
    loss_log = []
    loss_vae = []
    weight = []
    x = []
    div = 1./float(len(dataset) if not overfit else num_overfit)
    t1 = None
    
    
    
    """
    Actual training loop
    """
    for epoch in range(epochs):
        print("Starting Epoch %d/%d (%s seconds)"%(epoch+1,epochs, 'N/A' if t1 is None else '{:.2f}'.format(time.time()-t1)))
        t1 = time.time()
        for i, (data, target) in enumerate(dataset):
            if overfit:
                if i >= num_overfit:
                    break
            if cuda:
                data = data.cuda()
            loss = None
            optim.zero_grad()
            if type(model) == AE.VariationalAutoencoder:
                prediction, mean, log_var = model(data)
                loss_img = loss_fn(prediction, data)
                ## KL = sum_i sigma_i^2 + mu_i^2 - log(sigma_i) -1
                kl_div = 0.5*torch.sum(mean**2 +torch.exp(log_var) - log_var - 1.0)
                #loss = loss_img + kl_loss_weight*kl_div#/(2*torch.exp(2*kl_loss_weight))+kl_loss_weight
                l = weighted_loss([loss_img, kl_div])
                loss = sum(l)
                loss_vae.append([loss_img.detach().cpu(), kl_div.detach().cpu()])
                weight.append([weighted_loss.weight0.detach().cpu(), weighted_loss.weight1.detach().cpu()])
            else:
                prediction = model(data)
                loss = loss_fn(prediction, data)
            loss.backward()
            optim.step()
            loss_log.append(loss.detach().cpu())
            x.append(epoch+float(i)*div)
            if i % log_every == 0:
                print("  Loss after %d/%d iterations: %f"%(i,len(dataset) if not overfit else num_overfit,loss))
                if use_vis:
                    vis_plot = vis.line(win=vis_plot,X=x,Y=loss_log, env=vis_env, opts=plt_dict)
                    img = prediction.cpu()
                    img = img if img.shape[0] ==1 else img[0]
                    vis_image = vis.image(img,win=vis_image, env=vis_env)
                    if type(model) == AE.VariationalAutoencoder:
                        vis_vae = vis.line(win=vis_vae, X=x,Y=loss_vae, env=vis_env, opts=vae_dict)
                        vis_weights = vis.line(win=vis_weights, X=x, Y=weight, env=vis_env, opts=vae_w_dict)
                    vis.save(envs=[vis_env])
        if ((epoch + 1) % save_interval) == 0:
            save_model({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, save_dir, 'epoch_'+str(epoch+1))
    
        
    save_model({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict()}, save_dir, 'final_model')      
    return optim

def train_GAN(args, dataset, gen, disc, loss_fn, config, num_overfit=-1, resume_optim=None, input_size=(1, 28,28)):
    lr=config.getfloat('HYPERPARAMS', 'lr')
    print("Learning rate is %f"%lr)
    
    """
    Create optimizer
    """
    gen_opt_list = [{'params':gen.parameters()}]
    disc_opt_list = [{'params':disc.parameters()}]
    if config.get('HYPERPARAMS', 'disc_optimizer') == 'SGD':
        disc_optim = torch.optim.SGD(params=disc_opt_list, lr=lr)
    elif config.get('HYPERPARAMS', 'disc_optimizer') == 'Adam':
        disc_optim = torch.optim.Adam(params=disc_opt_list, lr=lr)
    else:
        raise NotImplementedError('Unknown optimizer!')
    if config.get('HYPERPARAMS', 'gen_optimizer') == 'SGD':
        gen_optim = torch.optim.SGD(params=gen_opt_list, lr=lr)
    elif config.get('HYPERPARAMS', 'gen_optimizer') == 'Adam':
        gen_optim = torch.optim.Adam(params=gen_opt_list, lr=lr)
    else:
        raise NotImplementedError('Unknown optimizer!')
    ##init model
    if resume_optim is not None:
        path, name = os.path.split(resume_optim)
        checkpoint_gen = torch.load(os.path.join(path, 'gen_{:s}').format(name))
        gen_optim.load_state_dict(checkpoint_gen['optimizer_state_dict'])
        gen.load_state_dict(checkpoint_gen['model_state_dict'])
        checkpoint_disc = torch.load(os.path.join(path, 'disc_{:s}'.format(name)))
        disc_optim.load_state_dict(checkpoint_disc['optimizer_state_dict'])
        disc.load_state_dict(checkpoint_disc['model_state_dict'])
        print("Model/Optimizer loaded")
        
        
    """
    Initialize training parameters
    """
    overfit = num_overfit > -1
    save_interval = config.getint('TRAINING', 'save_epoch_interval', fallback=10)
    ##create save folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    path = '{:s}_{:s}_{:s}'.format(args.model, args.data, timestamp)
    if overfit:
        path = '{:s}_overfitted'.format(path)
    save_dir = os.path.join('trained_models/', path)
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copyfile(args.config, save_dir+'/settings.config')
    except OSError:
        print ('Error: Creating directory. ' + save_dir)
    print("Created save directory at %s"%save_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("CUDA available")
        if resume_optim is not None:
            for state in disc_optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                             state[k] = v.cuda()
            for state in gen_optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                             state[k] = v.cuda()
    gen.to(device).train()
    disc.to(device).train()
    
    
    """
    Initialize visdom
    """
    use_vis = config.getboolean('TRAINING', 'visdom', fallback=True)
    if use_vis:
        print('Use visdom')
        vis_env = 'training'
        vis = visdom.Visdom()
        assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
        plt_dict = dict(name='loss', xlabel='epoch', ylabel='loss', legend=['gen loss', 'disc loss'])
        vis_plot = vis.line(X = [0, 0], Y=[0,0],env=vis_env,opts=plt_dict)
        vis_image = vis.image(np.zeros(input_size),env=vis_env)
    log_every = max(1, int(len(dataset)/config.getint('TRAINING', 'log_every_dataset_chunk', fallback=5))) if not overfit else num_overfit
    if 'epochs' not in config['HYPERPARAMS']:
        warnings.warn('Number of epochs not specified in config - Use 20 as default', Warning)
    epochs = config.getint('HYPERPARAMS', 'epochs', fallback=20)
    disc_log = []
    gen_log = []
    x = []
    div = 1./float(len(dataset) if not overfit else num_overfit)
    L = len(dataset) if not overfit else num_overfit
    t1 = None
    latent_dim = config.getint('HYPERPARAMS', 'latent_dim', fallback=10)
    flip_prob = config.getfloat('GAN_HACKS', 'flip_prob', fallback=0.9)
    noise_factor = config.getfloat('GAN_HACKS', 'noise_factor', fallback=0.1)
    noisy_labels = config.getboolean('GAN_HACKS', 'noisy_labels', fallback=False)
    input_noise = config.getboolean('GAN_HACKS', 'input_noise', fallback=False)
    
    
    """
    Actual training loop
    """
    for epoch in tqdm.tqdm(range(epochs), total=epochs, desc='Epochs'):
        #print("Starting Epoch %d/%d (%s seconds)"%(epoch+1,epochs, 'N/A' if t1 is None else str(time.time()-t1).format('%.1f')))
        t1 = time.time()
        for i, (data, target) in tqdm.tqdm(enumerate(dataset), total=L, leave=False):
            if overfit:
                if i >= num_overfit:
                    break
            data = data.to(device)
            gen_loss = None
            disc_loss = None
            gen_optim.zero_grad()
            
            data_size = data.size(0)
            
            ## fake batch
            fake_input_noise = torch.normal(torch.zeros(data_size, latent_dim), torch.ones(data_size, latent_dim)).to(device)
            fake_images = gen(fake_input_noise)
            fake_prediction = disc(fake_images)
            if noisy_labels:
                real_targets = torch.rand((data_size,1), device=device)*0.5+0.7
            else:
                real_targets = torch.ones((data_size,1), device=device)
            gan_loss = loss_fn(fake_prediction, real_targets)
            gan_loss.backward()
            gen_optim.step()
            
            ## real batch
            disc_optim.zero_grad()
            if noisy_labels:
                flip_vector = torch.rand((data_size, 1), device=device) > flip_prob
                real_targets = torch.rand((data_size,1), device=device)*0.5+0.7
                fake_targets = torch.rand((data_size,1), device=device)*0.3
                real_targets[flip_vector] -= 1.0
                fake_targets[flip_vector] += 1.0
            else:
                real_targets = torch.ones((data_size,1), device=device)
                fake_targets = torch.zeros((data_size,1), device=device) 
            if input_noise:
                noise = torch.randn(data.size(), device=device)*noise_factor
                data += noise
                fake_images += noise
            real_prediction = disc(data)
            disc_lossA = loss_fn(real_prediction, real_targets) 
            disc_lossA.backward()
            disc_optim.step()
            
            disc_optim.zero_grad()
            disc_lossB = loss_fn(disc(fake_images.detach()), fake_targets)
            disc_lossB.backward()
            disc_optim.step()
            
            disc_log.append(disc_lossA.detach().cpu() + disc_lossB.detach().cpu())
            gen_log.append(gan_loss.detach().cpu())
            x.append(epoch+float(i)*div)
            if i % log_every == 0:
                #print("  Loss after %d/%d iterations: %f"%(i,len(dataset) if not overfit else num_overfit,loss))
                if use_vis:
                    #x = np.asarray([0]) if i == 0 and epoch==0 else np.asarray(range(0,epoch*(len(dataset) if not overfit else num_overfit)+i+1))
                    vis_plot = vis.line(win=vis_plot,X=x,Y=disc_log, name='disc loss', env=vis_env, opts=plt_dict,update='append')
                    vis_plot = vis.line(win=vis_plot,X=x,Y=gen_log, name='gen loss', env=vis_env, opts=plt_dict,update='append')
                    x.clear()
                    disc_log.clear()
                    gen_log.clear()
                    img = fake_images[0].cpu()
                    img = img if img.shape[0] ==1 else img[0]
                    vis_image = vis.image(img,win=vis_image, env=vis_env)
                    vis.save(envs=[vis_env])
        if epoch % save_interval == 0 and epoch > 0:
            save_model(gen, gen_optim, save_dir, 'gen_epoch_{:s}'.format(str(epoch)))
            save_model(disc, disc_optim, save_dir, 'disc_epoch_{:s}'.format(str(epoch)))
    
        
    
    save_model(gen, gen_optim, save_dir, 'gen_final_model')
    save_model(disc, disc_optim, save_dir, 'disc_final_model')  
    return None

            
