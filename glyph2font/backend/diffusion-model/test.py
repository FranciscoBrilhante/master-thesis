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
from utils import create_dirs_logs, compute_fid, compute_mse, save_list_table, load_inputs_glyphnet
from data.diffusion_model import data_loader, save_random_batch_elem
from data.common import unfold_image, fold_image, load_image_to_tensor
from models.feature_extractor import GlyphGan
from models.diffusion_model import UNet


@torch.no_grad()
def sample(x: torch.Tensor, max_steps: int, model: UNet, content: torch.Tensor, style: torch.Tensor):
    # Remove noise for $T$ steps
    for t_ in range(max_steps):
        # $t$
        t = max_steps - t_ - 1
        # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        timestamps=x.new_full((x.size(0),), t, dtype=torch.long)
        x = model.p_sample(x, timestamps , content, style)

    return x

@torch.no_grad()
def test_diffusion_model(config):
    create_dirs_logs(config)
    path_log = os.path.join(config['path_log'], config['logname'])
    path_checkpoint = os.path.join(path_log, "Checkpoints", f"last.pt")
    path_data_test = config['test_data']
    device = config['device']

    model = load(path_checkpoint, map_location=torch.device(device))
    model.eval()


    data_test = data_loader(path=path_data_test, style_dir=config['style_dir_test'], batch_size=1, seed=config['seed'])
    for batch in tqdm(data_test):
        batch_x, targets, content, style, name, glyph_id = batch
        batch_x, targets, content, style = batch_x.to(config['device']), targets.to(config['device']), content.to(config['device']), style.to(config['device'])
        output=sample(torch.randn([1,1,64,64], device=config['device']),1000,model,content, style)
        
        path_to_save_image = Path(os.path.join(path_log, 'test', str(glyph_id[0].item()), name[0] ))
        path_to_save_image.parent.mkdir(exist_ok=True, parents=True)
        save_image(output[0], path_to_save_image)


def test_glyphnet(config, text_file_name):
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
        inputs = load_inputs_glyphnet(os.path.join(path_data_test, font_list[i]), device=device, n_glyphs_visible=n_glyphs_visible, glyphs_visible=glyphs_visible)
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


if __name__ == "__main__":
    pass
