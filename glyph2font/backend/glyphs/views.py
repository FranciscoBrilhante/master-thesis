from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from .models import GenerativeModel
from .forms import Generate
import string
import os
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.files.base import ContentFile
from io import BytesIO, StringIO
from pathlib import Path
import uuid
import glob
from tqdm import tqdm

# torch modules
import torch
from torch import load
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image

from zipfile import ZipFile
from dotenv import load_dotenv
from PIL import Image as ImagePIL

import sys
sys.path.insert(0, './gan')
sys.path.insert(0, './diffusion-model')


def index(request):
    return JsonResponse({"status": 200, "description": "Hello world. You're at the glyphs index."})


def listModels(request):
    if request.method == 'GET':
        models = GenerativeModel.objects.all()
        response = {'status': 200, 'models': [{
            'name': model.name,
            'description_en': model.description_en,
            'description_pt': model.description_en,
            'inputs': model.sugested_inputs,
            'eta': model.eta,
        } for model in models]}
        return JsonResponse(response)
    else:
        return JsonResponse({'status': 400})


def generate(request):
    if request.method == 'POST':
        form = Generate(request.POST, request.FILES)
        if form.is_valid():
            model = form.cleaned_data['model']
            format = form.cleaned_data['format']
            poligons = form.cleaned_data['poligons']
            segments = form.cleaned_data['segments']
            colorscheme = form.cleaned_data['colorscheme']
            labels = form.cleaned_data['labels']
            width = form.cleaned_data['width']
            height = form.cleaned_data['height']
            image = request.FILES["images"]

            image = ImagePIL.open(image)
            inputs = prepareInputs(image, labels, colorscheme, width, height)
            output = generateOutput(inputs, model)

            files_to_delete = []
            if format == 'svg':
                file_name, files_created = generateSVG(output, poligons, segments)
                for fil in files_created:
                    files_to_delete.append(fil)

            elif format == 'png':
                img_io = BytesIO()
                if colorscheme == 'white':
                    output=transforms.Lambda(lambda t: -t + 1)(output)

                transforms.ToPILImage()(output).save(img_io, format='PNG', quality=100)
                file_name = default_storage.save("outputs/image.png", ContentFile(img_io.getvalue(), 'imgage.png'))

            for fil in files_to_delete:
                default_storage.delete(fil)

            return JsonResponse({'status': 200, 'font': file_name})

        else:
            return JsonResponse({'status': 401})
    else:
        return JsonResponse({'status': 400})


@torch.no_grad()
def sample(x: torch.Tensor, max_steps: int, model: torch.nn.Module, content: torch.Tensor, style: torch.Tensor):
    # Remove noise for $T$ steps
    for t_ in range(max_steps):
        # $t$
        t = max_steps - t_ - 1
        # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        timestamps=x.new_full((x.size(0),), t, dtype=torch.long)
        x = model.p_sample(x, timestamps , content, style)

    return x

def prepareInputs(image, labels, color_scheme, width, height):
    labels = list(labels)
    abc = list(string.ascii_uppercase)

    trans_array = [
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ]
    if color_scheme == 'white':
        trans_array += [transforms.Lambda(lambda t: -t + 1)]
    trans = transforms.Compose(trans_array)

    image = trans(image)
    inputs = torch.ones((26, 64, 64))
    for i in range(len(labels)):
        aux=image[:, :, width*i:width*(i+1)].unsqueeze(0)
        aux=F.interpolate(aux, size=(64,64))
        inputs[abc.index(labels[i]), :, :] = aux
    inputs = inputs.unsqueeze(0).to('cpu')
    return inputs


def generateOutput(inputs, model_name):
    isDiffusion=GenerativeModel.objects.get(name=model_name).is_diffusion
    model_path = GenerativeModel.objects.get(name=model_name).log_path
    device="cpu"
    if isDiffusion:
        aux_path=GenerativeModel.objects.get(name=model_name).aux_diff_path
        diff_path=GenerativeModel.objects.get(name=model_name).diff_path

        print(inputs.size())
        print("loading aux model")
        aux_model = load(aux_path, map_location=torch.device(device))
        aux_model.eval()
        style = aux_model.encode(inputs)

        print("style extracted")
        print(style.size())

        print("loading main model")
        model = load(diff_path, map_location=torch.device(device))
        model.eval()
        
        final=torch.ones((1, 64, 1664)).to(device)
        for glyph in tqdm(range(26)):
            content=torch.Tensor([glyph]).long().unsqueeze(0).to(device)
            print(content.size())
            print(style.size())
            
            output=sample(torch.randn([1,1,64,64], device=device),1000,model,content, style)
            output=transforms.Lambda(lambda t: -t+1.)(output)
            final[0, :, glyph*64:(glyph+1)*64] = output
        return final
    
    else:
        model = load(model_path, map_location=torch.device('cpu'))
        model.eval()

        output = model.forward(inputs)
        output = unfold_image(output[0])
        return output


def unfold_image(tensor: Tensor):
    unfolded = torch.ones((1, 64, 1664)).to(tensor.device)
    for i in range(26):
        unfolded[0, :, i*64:(i+1)*64] = tensor[i, :, :]
    return unfolded


def generateSVG(outputs: Tensor, num_paths: int, num_segments: int):
    files_created = []
    zip_path = f'outputs/{str(uuid.uuid4())}.zip'
    # Path(zip_path).mkdir(exist_ok=True, parents=True)
    zipObj = ZipFile(zip_path, 'w')
    for i in range(26):
        img_io = BytesIO()
        transforms.ToPILImage()(outputs[0, :, i*64:(i+1)*64]).save(img_io, format='PNG', quality=100)

        png_file_name = default_storage.save(f"outputs/{str(i)}.png", ContentFile(img_io.getvalue(), 'imgage.png'))
        svg_file_name = f'{png_file_name[:-4]}.svg'
        files_created.append(png_file_name)
        files_created.append(svg_file_name)

        os.system(
            f"python live/LIVE/main.py --config live/LIVE/config/base.yaml --experiment main --output_path {svg_file_name} --target {png_file_name}  --num_paths={num_paths} --num_segments={num_segments} --num_iter 50")
        zipObj.write(svg_file_name)

    zipObj.close()
    return zip_path, files_created


"""     
def image_to_tensor(image:Image):
    trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    image = trans(image)
    return image

def fold_image(tensor: Tensor) -> Tensor:
    sample_channeled = torch.zeros((26, 64, 64))
    for i in range(26):
        sample_channeled[i, :, :] = tensor[0, :, 64*i:64*(i+1)]
    return sample_channeled
"""