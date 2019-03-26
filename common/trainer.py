import torch
import models.Autoencoder as AE
import visdom
import matplotlib.pyplot as plt
import datetime, time
import numpy as np
import os, sys
import gc
import shutil

def save_model(save_dict, save_dir, name):
    path = save_dir + '/'+name
    torch.save(save_dict, path)
    print('Saved model %s to %s'%(name, path))
    
    
def train_GAN(args, model, config, loader, loss, num_overfit=-1):
    if config['optimizer_generator'] == 'SGD':
        optim_gen = torch.optim.SGD(params=list(model.generator.parameters()), lr=float(config['lr_gen']))
    elif config['optimizer_generator'] == 'Adam':
        optim_gen = torch.optim.Adam(params=list(model.generator.parameters()), lr=float(config['lr_gen']))
    else:
        raise NotImplementedError('Optimizer not supported')
    if config['optimizer_discriminator'] == 'SGD':
        optim_dis = torch.optim.SGD(params=list(model.discriminator.parameters()), lr=float(config['lr_disc']))
    elif config['optimizer_discriminator'] == 'Adam':
        optim_dis = torch.optim.Adam(params=list(model.discriminator.parameters()), lr=float(config['lr_disc']))
    else: 
        raise NotImplementedError('Optimizer not supported')
    
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        
        
    ##create save folder
    save_interval = int(config['save_epoch_interval'])
    overfit = num_overfit > -1
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
    
    
        
    use_vis = 'visdom' in config.keys() and config['visdom'].lower() == 'true'
    if use_vis:
        print('Use visdom')
        vis_env = 'training'
        vis = visdom.Visdom()
        assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
        plt_dict = dict(name='training loss',ytickmax=10, xlabel='epoch', ylabel='loss', legend=['g_loss', 'd_loss'])
        vis_plot = vis.line(Y=[0],env=vis_env,opts=plt_dict)
        vis_image = vis.image(np.zeros((28,28)),env=vis_env)
        
    log_every = max(1, int(len(loader)/int(config['log_every_dataset_chunk']))) if not overfit else num_overfit
    epochs = int(config['epochs'])
    bs = int(config['batch_size'])
    t1 = None
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    losses = []
    x = []
    div = 1./float(len(loader) if not overfit else num_overfit)
    
    for epoch in range(epochs):
        print("Starting Epoch %d/%d (%s seconds)"%(epoch+1,epochs, 'N/A' if t1 is None else str(datetime.timedelta(seconds=time.time()-t1))))
        t1 = time.time()
        for i, (data, target) in enumerate(loader):
            if overfit:
                if i >= num_overfit:
                    break
            
            optim_gen.zero_grad()
            
            bs = data.size()[0] if not overfit else 1

            fake = model.generate(bs, cuda)
            real = data.cuda() if cuda else data
            target_real = Tensor(bs,1).fill_(1.0)
            target_fake = Tensor(bs,1).fill_(0.0)
            
            g_loss = loss(model.discriminate(fake), target_real)
            g_loss.backward()
            optim_gen.step()
            
            optim_dis.zero_grad()
            r_loss = loss(model.discriminate(real), target_real)
            f_loss = loss(model.discriminate(fake.detach()), target_fake)
            d_loss = 0.5*(r_loss+f_loss)
            d_loss.backward()
            optim_dis.step()
            
            losses.append([g_loss.detach().cpu(), d_loss.detach().cpu()])
            x.append(epoch+float(i)*div)
            
            if i % log_every == 0:
                #print("  Loss after %d/%d iterations: %f"%(i,len(dataset) if not overfit else num_overfit,loss))
                
                if use_vis:
                    vis_plot = vis.line(win=vis_plot,X=x,Y=losses, env=vis_env, opts=plt_dict)
                    img = fake.view(-1,1,28,28).cpu()
                    img = img if img.shape[0] ==1 else img[0]
                    vis_image = vis.image(img,win=vis_image, env=vis_env)
                    
            
        if ((epoch + 1) % save_interval) == 0:
            save_model({'model_state_dict': model.state_dict(), 'gen_optimizer_state_dict': optim_gen.state_dict(), 'dis_optimizer_state_dict': optim_dis.state_dict()}, save_dir, 'epoch_'+str(epoch+1))
            
            
            
    
    
    save_model({'model_state_dict': model.state_dict(), 'gen_optimizer_state_dict': optim_gen.state_dict(), 'dis_optimizer_state_dict': optim_dis.state_dict()}, save_dir, 'final_model')
    print('Done')
    

def train_AE(args, dataset, model, loss_fn, config, num_overfit=-1, resume_optim=None, input_size=(1, 28,28)):
    print("Learning rate is %f"%float(config['lr']))
    
    """
    Create optimizer
    """
    opt_list = list(model.parameters())
    if type(model) == AE.VariationalAutoencoder:
        #kl_loss_weight = torch.tensor(float(config['VAE_loss_weight']))
        weighted_loss = AE.WeightedMultiLoss(init_values=[float(config['IMG_loss_weight']), float(config['VAE_loss_weight'])], learn_weights=config['learn_loss_weights']=='True')
        opt_list = list(opt_list +list(weighted_loss.parameters()))
    if config['optimizer'] == 'SGD':
        optim = torch.optim.SGD(params=opt_list, lr=float(config['lr']))
    elif config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(params=opt_list, lr=float(config['lr']))
    else:
        print('Unknown optimizer!')
    ##init model
    #AE.init(model)
    if resume_optim is not None:
        checkpoint = torch.load(resume_optim)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model/Optimizer loaded")
        
    if not 'save_epoch_interval' in config.keys():
        raise KeyError('save_epoch_interval not given in config')
        
    """
    Initialize training parameters
    """
    overfit = num_overfit > -1
    save_interval = int(config['save_epoch_interval'])
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
    use_vis = 'visdom' in config.keys() and config['visdom'] == 'true'
    if use_vis:
        print('Use visdom')
        vis_env = 'training'
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
            
    log_every = max(1, int(len(dataset)/int(config['log_every_dataset_chunk']))) if not overfit else num_overfit
    epochs = int(config['epochs'])
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
            
