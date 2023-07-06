import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
import numpy
import string

# pytorch modules
import torch
from torch import load, nn, optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.nn import MSELoss

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

# local modules
from configs import *
from utils import create_dirs_logs, compute_fid, compute_mse, save_list_table, analize_table, plot_histogram, draw_red_square
from data.common import unfold_image, fold_image, load_image_to_tensor
from models import glyphgan as model


def load_inputs(source_path: str, device, n_glyphs_visible: int, glyphs_visible: list[int] | None):
    """
    :return: tensor width shape (channels,height, width)
    """
    image = Image.open(source_path)
    trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image = trans(image)
    image = image.to(device)
    image = fold_image(image)

    if glyphs_visible == None:
        idx_to_hide = random.sample(list(range(26)), 26-n_glyphs_visible)
    else:
        idx_to_hide = [i for i in range(26) if i not in glyphs_visible]

    for idx in idx_to_hide:
        image[idx, :, :] = 1

    image = image.unsqueeze(0).detach().to(device)
    return image


def test_default(config):
    create_dirs_logs(config)

    path_log = os.path.join(config['path_log'], config['logname'])
    path_data_test = config['data_path_test']
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")

    glyphs_visible = config.get('glyphs_visible', None)
    device = config['device']
    n_glyphs_visible = config['n_visible']

    model = load(path_checkpoint)
    model.eval()
    model.to(device)

    font_list = os.listdir(os.path.join(path_data_test))
    with torch.no_grad():
        for i in tqdm(range(len(font_list))):
            image = load_inputs(os.path.join(path_data_test, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
            output = model.forward(image)
            output = unfold_image(output[0])
            path_to_save_image = Path(os.path.join(path_log, 'test', font_list[i]))
            path_to_save_image.parent.mkdir(exist_ok=True, parents=True)
            save_image(output, path_to_save_image)


def test_with_metrics(config, text_file_name):
    create_dirs_logs(config)
    path_log = os.path.join(config['path_log'], config['logname'])
    path_data_test = config['data_path_test']
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")
    path_table = os.path.join(path_log, text_file_name)
    device = config['device']
    n_glyphs_visible = config['n_visible']
    glyphs_visible = config.get('glyphs_visible', None)

    model = load(path_checkpoint, map_location=torch.device(device))
    model.eval()

    font_list = os.listdir(os.path.join(path_data_test))
    data = []

    meanMSE = 0
    meanSSIM = 0
    for i in tqdm(range(len(font_list))):
        inputs = load_inputs(os.path.join(path_data_test, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
        output = model.forward(inputs)
        output = unfold_image(output[0])

        target = load_image_to_tensor(os.path.join(path_data_test, font_list[i]), unsqueeze=False, device=device)

        # mse
        mseLoss = MSELoss()
        mse = mseLoss(output, target).tolist()
        meanMSE += mse

        ssimLoss = SSIM(data_range=1.0)
        ssimLoss.update((output.unsqueeze(0), target.unsqueeze(0)))
        ssim = ssimLoss.compute()
        ssimLoss.reset()
        meanSSIM += ssim

        data.append([font_list[i][:-4], mse, ssim])

        path_to_save_image = Path(os.path.join(path_log, 'test', font_list[i]))
        path_to_save_image.parent.mkdir(exist_ok=True, parents=True)
        save_image(output, path_to_save_image)

    fid = compute_fid(config['data_path_test'], os.path.join(path_log, 'test'), batch_size=64, device=device)
    data = [[meanMSE/len(font_list), meanSSIM/len(font_list), fid]]+data
    save_list_table(path_table, data)


"""
def draw_visuals_exp3():
    
    to_remove=[10,7,1,13,17,3,15,14,18]
    for k in tqdm(range(3,10)):
        config=conf_default()
        config['path_log']= '../logs/experiment3'
        config['logname']=f'default_n_{k}_seed_0'
        config['seed']=0
        config['glyphs_visible']=[]
        for elem in list(range(26)):
            if elem in to_remove[0:k]:
                config['glyphs_visible'].append(elem)
        print(config['glyphs_visible'])
        config['n_visible']=k
    
        get_distribution(config,text_file_name='data.txt',save_outputs=True)
    
    device="cuda:0"
    image=torch.ones(size=(3,64,1664),device=device, requires_grad=False)
    filenames=os.listdir('/home/francisco/mc-gan/datasets/Capitals64/test')

    to_remove=[10,7,1,13,17,3,15,14,18]
    
    for name in tqdm(random.sample(filenames,10)):
        parcel=load_image_to_tensor(os.path.join('/home/francisco/mc-gan/datasets/Capitals64/test', name), unsqueeze=False, device=device)
        parcel=torch.cat((parcel,parcel,parcel),dim=0)
        image=torch.cat((image,parcel),dim=1)
    
        for k in range(3,10):
            parcel=load_image_to_tensor(os.path.join(f'/home/francisco/logs/experiment3/default_n_{k}_seed_0/test', name), unsqueeze=False, device=device)
            parcel=torch.cat((parcel,parcel,parcel),dim=0)
            for elem in to_remove[0:k]:
                parcel=draw_red_square(parcel,startX=0,startY=64*elem,width=64)
           
            image=torch.cat((image,parcel),dim=1)
        
        image=torch.cat((image,torch.ones(size=(3,64,1664),device=device)),dim=1)

    save_image(image, f'/home/francisco/logs/experiment3/test.png')
    

if __name__ == "__main__":
    #draw_visuals_exp3()
    for seed in tqdm(range(0,15)):
        config=conf_experiment1_var1()
        config['logname']=f'default_seed_{seed}'
        config['seed']=seed
        config['device']='cpu'

        get_distribution(config,text_file_name='data.txt',save_outputs=True)
        raise Exception
        
        analize_table(path_log=f'/home/francisco/logs/experiment1/default_seed_{seed}', filename=f'{letter}.txt',tails_size=10, original_dataset='/home/francisco/mc-gan/datasets/Capitals64/test',generated_dataset=f'/home/francisco/logs/experiment1/default_seed_{seed}/test')
        f=open('/home/francisco/logs/combined_seed_0/data.txt','r')
        lines=[line.strip().split('\t') for line in f.readlines()]
        data=[float(line[1]) for line in lines]
        plot_histogram('/home/francisco/logs/combined_seed_0/histogram.png',data,50,title='Histograma de valores MSE sobre fontes de teste',xlegend='MSE',ylegend='Frequência')
        

    
"""
