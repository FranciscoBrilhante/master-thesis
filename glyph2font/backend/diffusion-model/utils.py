from torch.utils.tensorboard import SummaryWriter

import os
import torch
import json
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image

from torch import load
from torchvision import transforms
from torch.nn import MSELoss

# local modules
from data.common import load_image_to_tensor,unfold_image,fold_image

def create_dirs_logs(config: dict):
    log_dir = os.path.join(config['path_log'], config['logname'])
    for path_target in ["", 'test', 'Checkpoints']:
        Path(os.path.join(log_dir, path_target)).mkdir(exist_ok=True, parents=True)
    with open(f"{log_dir}/config.json", 'w') as fp:
        json.dump(config, fp,sort_keys=True, indent=2)


def update_telemetry(writer: SummaryWriter, t, step: int, loss: torch.tensor):
    loss_value = round(loss.item(), 5)
    t.set_description(f"Loss: {loss_value}")
    writer.add_scalar('loss/train', loss_value, step)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_list_table(path_file:str ,list:list):
    with open(path_file,'w') as f:
        for elem in list:
            for col in elem:
                f.write(f'{col}\t')
            f.write('\n')

def draw_red_square(image, startX:int , startY:int, width:int):
    image[0,startX:startX+width,startY:startY+2]=1
    image[1,startX:startX+width,startY:startY+2]=0
    image[2,startX:startX+width,startY:startY+2]=0

    image[0,startX:startX+width,startY+width-2:startY+width]=1
    image[1,startX:startX+width,startY+width-2:startY+width]=0
    image[2,startX:startX+width,startY+width-2:startY+width]=0

    image[0,startX:startX+2,startY:startY+width]=1
    image[1,startX:startX+2,startY:startY+width]=0
    image[2,startX:startX+2,startY:startY+width]=0

    image[0,startX+width-2:startX+width,startY:startY+width]=1
    image[1,startX+width-2:startX+width,startY:startY+width]=0
    image[2,startX+width-2:startX+width,startY:startY+width]=0

    return image

def compute_mse(real_folder: str, fake_folder: str, class_folders: bool = True, fake_suffix: bool = True) -> float:
    mean_mse = 0
    total = 0

    if class_folders:
        font_list = os.listdir(os.path.join(real_folder, "0"))
        for i in tqdm(range(len(font_list))):
            for j in range(26):
                real_image = load_image_to_tensor(os.path.join(real_folder, f'{j}', font_list[i]))
                fake_image = load_image_to_tensor(os.path.join(fake_folder, f'{j}', font_list[i]))

                loss = MSELoss()
                out = loss(fake_image, real_image)
                mean_mse += out
                total += 1
    else:
        font_list = os.listdir(real_folder)
        for i in tqdm(range(len(font_list))):
            real_path = os.path.join(real_folder, font_list[i])
            if fake_suffix:
                fake_path = os.path.join(fake_folder, font_list[i][:-4]+"_fake_B.png")
            else:
                fake_path = os.path.join(fake_folder, font_list[i])
            real_image = load_image_to_tensor(real_path)
            fake_image = load_image_to_tensor(fake_path)

            loss = MSELoss()
            out = loss(fake_image, real_image).tolist()
            mean_mse += out
            total += 1

    return mean_mse/total


def compute_fid(real_folder: str, fake_folder: str, batch_size: int, device:str) -> float:
    stream = os.popen(f'python -m pytorch_fid {real_folder} {fake_folder} --batch-size {batch_size} --device {device}')
    output = stream.read()
    return float(output.strip().split(':')[1].strip())


def load_inputs_glyphnet(source_path: str, device, n_glyphs_visible: int, glyphs_visible: list[int] | None):
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


def encode_features(config, source_folder_path, target_folder_path):
    create_dirs_logs(config)
    path_log = os.path.join(config['path_log'], config['logname'])
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")
    device = config['device']
    n_glyphs_visible = config['n_visible']
    glyphs_visible = config.get('glyphs_visible', None)

    model = load(path_checkpoint, map_location=torch.device(device))
    model.eval()

    for split_name in ['train', 'test', 'val']:
        folder_path=os.path.join(source_folder_path, split_name)
        Path(target_folder_path, split_name).mkdir(exist_ok=True, parents=True)
        font_list = os.listdir(folder_path)
        
        for i in tqdm(range(len(font_list))):
            inputs = load_inputs_glyphnet(os.path.join(folder_path, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
            features = model.encode(inputs).squeeze(0)
            torch.save(features, os.path.join(target_folder_path,split_name,f'{font_list[i][:-4]}.pt'))

