#!/bin/bash

echo "Cloning LIVE-svg"
git clone https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization.git live
cd live/DiffVG
git submodule update --init --recursive
python setup.py install
cd ../..
mv main.py live/LIVE/main.py

echo "Initializing Django app and server"
rm -r -f glyphs/migrations/*.py
rm db.sqlite3
python manage.py makemigrations 
python manage.py migrate 
python manage.py makemigrations glyphs 
python manage.py migrate glyphs 
python manage.py loaddata models 