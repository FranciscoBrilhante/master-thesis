FROM ubuntu:20.04

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update -y && apt-get upgrade -y && apt install build-essential -y && apt install wget -y && apt install git -y

# install miniconda
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
RUN conda init bash

WORKDIR /home/backend

RUN echo "Installing LIVE-svg dependencies" && \
    conda create -y -n live python=3.7 && \
    conda activate live && \
    conda install -y pytorch torchvision -c pytorch && \
    conda install -y numpy scikit-image && \
    conda install -y -c anaconda cmake && \
    conda install -y -c conda-forge ffmpeg && \
    pip install svgwrite svgpathtools cssutils numba torch-tools scikit-fmm easydict visdom && \
    pip install  opencv-python==4.5.4.60 && \
    echo "Installing Django app dependencies" && \
    conda install -y Django && \
    pip install django-cors-headers && \
    pip install Pillow && \
    pip install python-dotenv && \
    pip install requests && \
    pip install python-dotenv && \
    conda list

COPY . .

RUN conda activate live && \
    echo "Cloning LIVE-svg" && \
    git clone https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization.git live && \
    cd live/DiffVG && \
    git submodule update --init --recursive && \
    python3 setup.py install && \
    cd ../.. && \
    mv main.py live/LIVE/main.py && \
    mv base.yaml live/LIVE/config/base.yaml && \
    echo "Initializing Django app and server" && \
    rm -rf glyphs/migrations/*.py || true && \
    rm db.sqlite3 && \
    python3 manage.py makemigrations && \
    python3 manage.py migrate && \ 
    python3 manage.py makemigrations glyphs && \
    python3 manage.py migrate glyphs && \
    python3 manage.py loaddata models

EXPOSE 8000
CMD /bin/bash
# docker build -t glyph2font-backend .
# docker run -ti -p 8000:8000 --name glyph2font-backend glyph2font-backend
# docker rm glyph2font-backend

# conda activate live
# python3 manage.py runserver 0.0.0.0:8000