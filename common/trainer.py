import torch
import models.Autoencoder as AE
import visdom
import matplotlib.pyplot as plt
import numpy as np

def train(dataset, model, loss_fn, config, num_overfit=-1):
    print("Learning rate is %f"%float(config['lr']))
    if config['optimizer'] == 'SGD':
        optim = torch.optim.SGD(list(model.parameters()), lr=float(config['lr']))
    elif config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(list(model.parameters()), lr=float(config['lr']))
    else:
        print('Unknown optimizer!')
    cuda = torch.cuda.is_available()
    if cuda:
        print("CUDA available")
        model.cuda()
    use_vis = 'visdom' in config.keys() and config['visdom'] == 'true'
    if use_vis:
        print('Use visdom')
        vis_env = 'training'
        vis = visdom.Visdom()
        assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
        plt_dict = dict(name='training loss',ymin=0, xlabel='iterations', ylabel='loss', legend=['train_loss'])
        vis_plot = vis.line(Y=[0],env=vis_env,opts=plt_dict)
        vis_image = vis.image(np.zeros((1,28,28)),env=vis_env)
        if type(model) == AE.VariationalAutoencoder:
            vae_dict = dict(ymin=0, ymax=10, legend=['reconstruction loss', 'kl loss'])
            vis_vae = vis.line(Y=[0,0], env=vis_env, opts=vae_dict)
    overfit = num_overfit > -1
    log_every = int(len(dataset)/int(config['log_every_dataset_chunk'])) if not overfit else num_overfit
    epochs = int(config['epochs'])
    loss_log = []
    loss_vae = []
    for epoch in range(epochs):
        print("Starting Epoch %d/%d"%(epoch+1,epochs))
        for i, (data, target) in enumerate(dataset):
            if overfit:
                if i >= num_overfit:
                    break
            if cuda:
                data = data.cuda()
            loss = None
            if type(model) == AE.VariationalAutoencoder:
                prediction, mean, log_var = model(data)
                loss_img = loss_fn(prediction, data)
                ## KL = sum_i sigma_i^2 + mu_i^2 - log(sigma_i) -1
                kl_div = 0.5*torch.sum(mean**2 +torch.exp(log_var) - log_var - 1.0)
                loss = 10*loss_img + kl_div
                loss_vae.append([loss_img.detach().cpu(), kl_div.detach().cpu()])
            else:
                prediction = model(data)
                loss = loss_fn(prediction, data)
            loss.backward()
            optim.step()
            loss_log.append(loss.detach().cpu())
            if i % log_every == 0:
                print("  Loss after %d/%d iterations: %f"%(i,len(dataset) if not overfit else num_overfit,loss))
                if use_vis:
                    x = np.asarray([0]) if i == 0 and epoch==0 else np.asarray(range(0,epoch*(len(dataset) if not overfit else num_overfit)+i+1))
                    y = np.asarray(loss_log)
                    vis_plot = vis.line(win=vis_plot,X=x,Y=loss_log, env=vis_env, opts=plt_dict)
                    img = prediction.cpu()
                    img = img if img.shape[0] ==1 else img[0]
                    vis_image = vis.image(img,win=vis_image, env=vis_env)
                    if type(model) == AE.VariationalAutoencoder:
                        vis_vae = vis.line(win=vis_vae, X=x,Y=loss_vae, env=vis_env, opts=vae_dict)
                    vis.save(envs=[vis_env])
        
          
    return optim
            
