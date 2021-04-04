import os
import numpy as np
import argparse

import torch

import pickle
from PIL import Image

from utils import resnet50_feature_extractor, transform, int2cat

MODEL_NAME = os.environ.get("MODEL_NAME")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PRED_DATA_DIR = os.environ.get('PRED_DATA_DIR')

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--load_dir", type=str, default=CURRENT_DIR)
    return parser

def main():

    parser = _setup_parser()
    args = parser.parse_args()

    #load resnet model
    resnet_model = resnet50_feature_extractor(pretrained=True)

    #load logmodel
    with open(CURRENT_DIR+'/'+args.model_name, 'rb') as f:
        logmodel = pickle.load(f)
        
    # load images
    # get image from PRED_DATA for now
    img_file = '5.jpg' # for example
    img = Image.open(PRED_DATA_DIR+'/'+img_file)
    img = img.convert(mode='RGB')
    x = transform(img)
    x = torch.unsqueeze(x, dim=0)
    features = resnet_model(x)
    features = features.detach().numpy()

    # predict
    prediction = logmodel.predict(features)
    prediction = int2cat[prediction]

    # need pass back somewhere...

if __name__ == "__main__":
    main()