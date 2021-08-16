# waffle-removal

This repository contains a reference implementation for [WAFFLE](https://arxiv.org/abs/2008.07298) and also for adaptive adversaries trying to remove watermarkS during federated learning (FL). 

## Setup
Clone this repository and install necessary Python packages using conda environment. The necessary packages are also listed in the requirements.txt. However, we strongly suggest that you should run your codes using conda environment

1. Create the environment from the pysyft.yml file:
`conda env create -f pysyft.yml`
2. Activate the new environment: 
`conda activate pysyft`

We use PySyft 0.2.x. Its codebase is in its own branch [here](https://github.com/OpenMined/PySyft/tree/syft_0.2.x), but OpenMined announced that they will not offer official support for this version range.
## Training FL models

1. Train an FL model without watermarking: (The config folder named as **FL** contains configuration files for this purpose.)

`python main.py --config_file configurations/FL/mnist/0.ini --experiment training`

2. Train an FL model with watermarking: (The config folder named as **PR** contains configuration files for this purpose. Config files also include which watermark set is going to be used for watermarking)

`python main.py --config_file configurations/PR/cifar10/0.ini --experiment training`

## Training WR models

1. Train an WR model during training: (The config folder named as **WR** contains configuration files for this purpose.)

`python main.py --config_file configurations/WR/cifar10/100_20/1.ini --experiment training`

1. Train an WR model after training: (The config folder named as **WR/after_train/** contains configuration files for this purpose.)

`python main.py --config_file configurations/WR/after_train/mnist/0.ini --experiment training`

