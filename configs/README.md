# Format of configs
configs are in ConfigParser style

There are three sections with different sub parameters
### Hyperparams
 - batch_size: Defines size of batches during training (Needs to be at least 2 when batchnorm is used, if overfitting is used it is set to overfit parameter)
 - lr: learning rate; defines step size of optimization steps
 - epochs: Number of epochs that model will be trained
 - loss: Name of loss function that should be applied (Options: 'L1', 'L2', 'MSE', 'BCE', 'CrossEntropy')
#### GAN specific
 - gen_optimizer: Optimizer for generator (Options: 'SGD', 'Adam')
 - disc_optimizer: Optimizer for discriminator (Options: 'SGD', 'Adam')
 - latent_dim: Size of noise dimension used as input for generator
#### Autoencoder specific
 - hidden_dim_size: Dimension of encoding
 - encode_factor: resolution of input will be divided by 2^encode_value but channels increased by this factor
 - base_channels: Number of channels that autoencoder model will start with
 - IMG_loss_weight: Weight for loss of image reconstruction for variational autoencoder
 - VAE_loss_weight: Weight for KL-Divergence for variational autoencoder
 - learn_loss_weights: Train weights for variational autoencoder loss
 
### Training
 - visdom: Whether to use a visdom server (Highly recommended: start with python -m visdom.server)
 - input_size: size of input images in format NxM where N and M are dimensions of image
 - RGB: whether input images are RGB or one channel images
 - save_epoch_interval: Number of epochs after which an intermediate save of models is performed
 - log_every_dataset_chunk: How often visdom server is updated in 1/x epochs. E.g. for 5 server is updated every 20% of an epoch.
 
### GAN_HACKS
Several GAN Hacks from [repo](https://github.com/soumith/ganhacks) are implemented and can be set here.
 - noisy_labels: Whether or not to use labels that are sampled from a gaussian around the target or precise labels. Also allows label flips for discriminator training
 - flip_factor: Only if noisy_labels is allowed. Chance, that labels for discriminator are <b>correct</b>. E.g. for 0.9 10% of labels are incorrect.
 - input_noise: Add noise on input for discriminator
 - noise_factor: Only if input_noise is allowed. Factor that noise sampled from uniform gaussian is scaled with. Remember that input should be in [-1, 1].