#!/bin/bash

python -m venv venv &&
source venv/bin/activate &&
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 &&
pip install -r requirements.txt