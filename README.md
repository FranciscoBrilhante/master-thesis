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



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
