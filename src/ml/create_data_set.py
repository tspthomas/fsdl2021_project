import os
import numpy as np
import argparse

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
# from torchsummary import summary

from PIL import Image

from utils import resnet50_feature_extractor, transform, img_classes, cat2int

TRAIN_DATA_DIR = os.environ.get('TRAIN_DATA_DIR')
TEST_DATA_DIR = os.environ.get('TEST_DATA_DIR')
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_data_name", type=str, default="train")
    parser.add_argument("--in_data_dir", type=str, default=TRAIN_DATA_DIR)
    parser.add_argument("--save_dir", type=str, default=CURRENT_DIR+"/np_data/")
    return parser

def main():

    parser = _setup_parser()
    args = parser.parse_args()

    # load resnet model
    resnet_model = resnet50_feature_extractor(pretrained=True)

    # loop through DATA_DIR
    # create np arrays of img_file, feature_vector, cat2int[img_class]
    img_file_data = []
    feature_vector_data = []
    img_class_data = []
    for img_class in img_classes:
        for img_file in os.listdir(args.in_data_dir+img_class):
            img = Image.open(args.in_data_dir+img_class+'/'+img_file)
            img = img.convert(mode='RGB')
            x = transform(img)
            x = torch.unsqueeze(x, dim=0)
            features = resnet_model(x)

            img_file_data.append(img_class+'/'+img_file)
            feature_vector_data.append(features.data[0].numpy())
            img_class_data.append(cat2int[img_class])
    
    # save np arrays
    np.save(args.save_dir+args.out_data_name+"_img_file", img_file_data)
    np.save(args.save_dir+args.out_data_name+"_feature_vector", feature_vector_data)
    np.save(args.save_dir+args.out_data_name+"_img_class", img_class_data)

    
if __name__ == "__main__":
    main()