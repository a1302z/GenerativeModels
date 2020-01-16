# Format of configs
configs are in [ConfigParser](https://docs.python.org/3/library/configparser.html) style

There are three sections with different sub parameters
### Hyperparams
 - batch_size: Defines size of batches during training (Needs to be at least 2 when batchnorm is used, if overfitting is used it is set to overfit parameter)
 - lr: learning rate; defines step size of optimization steps
 - epochs: Number of epochs that model will be trained
 - loss: Name of loss function that should be applied (Options: 'L1', 'L2', 'MSE', 'BCE', 'CrossEntropy')
 - beta1, beta2: If adam is used as optimizer these beta values are additional hyperparams that can be set.
##### GAN specific
 - gen_optimizer: Optimizer for generator (Options: 'SGD', 'Adam')
 - disc_optimizer: Optimizer for discriminator (Options: 'SGD', 'Adam')
 - latent_dim: Size of noise dimension used as input for generator
 - GAN_noise_strength: Factor to multiply noise input for generator sampled from normal distribution with.
##### Autoencoder specific
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
 - log_every_dataset_chunk: How often visdom server is updated in 1/x epochs. E.g. for 5 server is updated every 20% of an epoch. (Visdom updates are slow!)
 
### GAN_HACKS
Several GAN Hacks from this [repo](https://github.com/soumith/ganhacks) are implemented and can be set here.
 - noisy_labels: Whether or not to add gaussian noise to labels.
 - label_noise_strength: factor for normal distribution added to target labels
 - label_flips: allow label flips for discriminator training
 - flip_factor: Chance, that labels for discriminator are <b>correct</b>. E.g. for 0.9 10% of labels are incorrect.
 - input_noise: Add noise on input for discriminator
 - noise_factor: Only if input_noise is allowed. Factor that noise sampled from uniform gaussian is scaled with. Remember that input should be in [-1, 1].