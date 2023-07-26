
import os
import random
import math
# pytorch modules
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision import transforms

#local modules
from .common import load_image_to_tensor, fold_image
    
class CapitalsDatasetDiffusion(Dataset):
    def __init__(self, root_dir, style_dir) -> None:
        super().__init__()
        self.root_dir=root_dir
        self.style_dir=style_dir
        self.total_fonts=len(os.listdir(self.root_dir))

    def __len__(self):
        return 26*self.total_fonts
        #return self.each_class_dimension

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        font_it_belongs=math.floor(idx/26)
        idx_inside_font=idx-font_it_belongs*26

        name=os.listdir(self.root_dir)[font_it_belongs]
        font_path=os.path.join(self.root_dir,os.listdir(self.root_dir)[font_it_belongs])
        
        target=load_image_to_tensor(font_path,unsqueeze=False,device='cpu')
        target=fold_image(target)
        target=target[idx_inside_font].unsqueeze(0)
        target=transforms.Lambda(lambda t: -t+1.)(target)
        target=transforms.Lambda(lambda t: (t*2.-1.))(target)

        model_input = torch.clone(target)

        content=torch.Tensor([idx_inside_font]).long()
        style_path=os.path.join(self.style_dir,os.listdir(self.root_dir)[font_it_belongs][:-4]+'.pt')
        style=torch.load(style_path, map_location='cpu')

        return model_input.detach(), target.detach(), content.detach(), style.detach(), name, idx_inside_font

def data_loader(path: str, style_dir:str, batch_size: int, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    data = CapitalsDatasetDiffusion(root_dir=path, style_dir=style_dir)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4, generator=g)
    return data_loader


def save_random_batch_elem(input_batch: tuple, output_batch: torch.Tensor , path: str):
    idx = random.randint(0,input_batch[0].size(0))
    model_input, targets = input_batch
    
    save_image(model_input[idx], os.path.join(path, "input.png"))
    save_image(targets[idx], os.path.join(path, "output.png"))
    save_image(output_batch[idx], os.path.join(path, "target.png"))