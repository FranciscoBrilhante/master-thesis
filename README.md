<a name="readme-top"></a>

<!-- ABOUT -->
## About

This repository contains all the code and data required to replicate all the experiments found in my Master Thesis.
* `datasets` contains all the datasets used (see bellow how to build this folder)
* `diffusion-model` contains the code for training and testing the diffusion model for glyph generation
* `gan` has all the code for training and testing GlyphNet. Adaped from [MC-GAN](https://github.com/zhourunlong/mc-gan)
* `glyph2font` is a web application that allows users to generate new fonts when a few prompt glyphs are provided.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
First, download the dataset from [here](https://drive.google.com/file/d/1cFDayTF2Gwh98AU_CYRcEc4oJ9surDUP/view?usp=sharing) and extract to the `datasets` folder.
```sh
tar xvfz ./datasets.tar.gz ./datasets
```

Install the following python modules:
```sh
pytorch
torchvision
pyTorch-ignite
scipy
pillow
matplotlib
tqdm
numpy
```

### Experiment Replication
Both `gan` and `diffusion-model` contain a `configs.py` file with configurations for all the experiments.
In order to train a given netowrk call `train` function with a config setup. <br></br>
Scripts to run these experiments can be found in `gan/experiments.py` and `diffusion-model/experiments.py`.
```python
#assuming current directory is ./gan
from configs import *
from train import train_default

config=conf_experiment1_var1()
train_default(config)
```
This will start trainning and saving a  model checkpoint in `/gan/logs` folder. This behaviour can me modified in the config file.

After obtaining a trained model, it can be used to generate test images  and compute performance metrics.
```python
#assuming current directory is ./gan
from configs import *
from test import test_default, test_with_metrics

config=conf_experiment1_var1()
test_with_metrics(config,'data.txt')
```
Metrics will be logged in `gan/logs/exp1/data.txt` while the generated outputs will be available in `gan/logs/exp1/test`
These procedures are the same for the diffusion model. In order to train a diffusion-model, a pre-trained style encoder is needed and referenced in the config file.
```python
from configs import *
from train train_glyphnet

config=conf_glyphnet()
train_glyphnet(config)
```

## Web Application
The web application is composed of a backend built with Django and a frontend built with NextJs
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Acknowledgements

This repository uses data and code adapted from [MC-GAN](https://github.com/zhourunlong/mc-gan) and [LabML](https://nn.labml.ai/diffusion/ddpm/index.html).
<p align="right">(<a href="#readme-top">back to top</a>)</p>
