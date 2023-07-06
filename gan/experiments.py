import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
import numpy
import string
from piqa import SSIM

#pytorch modules
import torch
from torch import load
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import MSELoss

#local modules
from configs import *
from utils import create_dirs_logs, compute_fid, compute_mse,save_list_table, analize_table, plot_histogram, draw_red_square
from data.common import unfold_image, fold_image, load_image_to_tensor
from models import glyphgan as model
from train import train_default
from test import test_default, test_with_metrics

#--------- Train models
"""
for seed in range(0,15):
        config=conf_experiment1_var1()
        config['logname']=f'default_seed_{seed}'
        config['seed']=seed
        train_default(config)
for seed in range(0,15):
        config=conf_experiment1_var2()
        config['logname']=f'mask_seed_{seed}'
        config['seed']=seed
        train_default(config)
for seed in range(0,15):
        config=conf_experiment1_var3()
        config['logname']=f'skeleton_seed_{seed}'
        config['seed']=seed
        train_default(config)


#--------- Generate Test Samples
#Compute MSE and SSIM for each output alongside the overall mean and FID on whole dataset

for seed in range(0,15):
        for letter in list(string.ascii_uppercase):
                config=conf_experiment1_var1()
                config['logname']=f'default_seed_{seed}'
                config['seed']=seed
                config['glyphs_visible']=[list(string.ascii_uppercase).index(letter)]
                test_with_metrics(config,f'{letter}.txt')

for seed in range(0,15):
        for letter in list(string.ascii_uppercase):
                config=conf_experiment1_var2()
                config['logname']=f'mask_seed_{seed}'
                config['seed']=seed
                config['glyphs_visible']=[list(string.ascii_uppercase).index(letter)]
                test_with_metrics(config,f'{letter}.txt')

for seed in range(0,15):
        for letter in list(string.ascii_uppercase):
                config=conf_experiment1_var3()
                config['logname']=f'skeleton_seed_{seed}'
                config['seed']=seed
                config['glyphs_visible']=[list(string.ascii_uppercase).index(letter)]
                test_with_metrics(config,f'{letter}.txt')




config=conf_experiment4_var1()
train_default(config)

config=conf_experiment4_var2()
train_default(config)


#--------- Generate Test Samples
#Compute MSE and SSIM for each output alongside the overall mean and FID on whole dataset

for seed in range(0,15):
        config=conf_experiment2_var1()
        config['logname']=f'subset_1_seed_{seed}'
        config['seed']=seed
        test_with_metrics(config,'data.txt')

for seed in range(0,15):
        config=conf_experiment2_var2()
        config['logname']=f'subset_2_seed_{seed}'
        config['seed']=seed
        test_with_metrics(config,'data.txt')


#--------- Generate Test Samples
#Compute MSE and SSIM for each output alongside the overall mean and FID on whole dataset

for seed in range(0,5):
        for k in range(3,10):
                config=conf_experiment3()
                config['logname']=f'default_n_{k}_seed_{seed}'
                config['n_visible']=k
                config['glyphs_visible']=None
                config['seed']=seed
                test_with_metrics(config,'data.txt')


#--------- Generate Test Samples
#Compute MSE and SSIM for each output alongside the overall mean and FID on whole dataset


config=conf_experiment4_var1()
test_with_metrics(config,'data.txt')


config=conf_experiment4_var2()
test_with_metrics(config,'data.txt')

config=conf_experiment4_var3()
train_default(config)

config=conf_experiment4_var3()
test_with_metrics(config,'data.txt')

for seed in range(5,15):
        config=conf_experiment4_var1()
        config['seed']=seed
        config['logname']=f'var1_seed_{seed}'
        train_default(config)
        #test_with_metrics(config,'data.txt')

for seed in range(6,15):
        config=conf_experiment4_var2()
        config['seed']=seed
        config['logname']=f'var2_seed_{seed}'
        train_default(config)
        #test_with_metrics(config,'data.txt')


for seed in range(0,15):
        config=conf_experiment4_var1()
        config['seed']=seed
        config['logname']=f'var1_seed_{seed}'
        #train_default(config)
        test_with_metrics(config,'data.txt')
"""
for seed in range(0,15):
        config=conf_experiment4_var2()
        config['seed']=seed
        config['logname']=f'var2_seed_{seed}'
        #train_default(config)
        test_with_metrics(config,'data.txt')


