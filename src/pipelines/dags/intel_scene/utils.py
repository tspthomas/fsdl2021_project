import os

from PIL import Image

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision.models import resnet 

import numpy as np

img_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
cat2int = {'buildings': 0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
int2cat = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
included_extensions = ['jpg','jpeg','png']

FEEDBACK_DATA_DIR = os.environ.get('FEEDBACK_DATA_DIR')
RAW_DATA_DIR = os.environ.get('RAW_DATA_DIR')
PROCESSED_DATA_DIR = os.environ.get('PROCESSED_DATA_DIR')

class Data():
    def __init__(self):
        self.seen_features = {}
        self.X = []
        self.y = []

class Resnet50Features(resnet.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_feat = torch.flatten(x, 1)

        return x_feat

def resnet50_feature_extractor(pretrained=False, **kwargs):
    model = Resnet50Features(resnet.Bottleneck,
                             [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50'], model_dir = '/opt/airflow/.cache/torch'))
    model.eval()
    return model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                normalize
                            ])

def extract_features_to_data(dataset_name, preload_dict=False):
    '''
    dataset_name (str): train / test
    preload_dict (bool): preload or not

    returns: (Data)
    '''
    # load resnet model
    resnet_model = resnet50_feature_extractor(pretrained=False)

    data = Data()

    if preload_dict:
        with open(os.path.join(RAW_DATA_DIR ,'processed', f'{dataset_name}.pickle'), 'rb') as f:
            data = pickle.load(f)

    for img_class in img_classes:
    
        data_dir = os.path.join(RAW_DATA_DIR, 
                    f'intel_image_scene/seg_{dataset_name}/seg_{dataset_name}', img_class)

        # Hack to account for non-images in the folder
        img_files = [fn for fn in os.listdir(data_dir) 
                        if any(fn.endswith(ext) for ext in included_extensions)]

        for img_file in img_files:

            img_path = os.path.join(data_dir, img_file)

            if img_path not in data.seen_features.keys():

                img = Image.open(img_path)
                img = img.convert(mode='RGB')
                x = transform(img)
                x = torch.unsqueeze(x, dim=0)
                features = resnet_model(x)

                data.seen_features[img_path] = True
                data.X.append(features.data[0].numpy())
                data.y.append(cat2int[img_class])

        print("{}: {} processed".format(dataset_name, img_class))

    return data

# def create_np_arrays(in_data, out_dataset_name):
#     '''
#     in_data (str): raw / feedback
#     out_dataset_name (str): train / test
#     '''

#     # load resnet model
#     resnet_model = resnet50_feature_extractor(pretrained=True)

#     # # loop through data_dir
#     # # create np arrays of img_file, feature_vector, cat2int[img_class]
#     img_filename_data = []
#     feature_vector_data = []
#     img_class_data = []
#     for img_class in img_classes:
    
#         data_dir = os.path.join(DATA_DIR, in_data, 
#                     'intel_image_scene/seg_{}/seg_{}'.format(out_dataset_name,out_dataset_name), img_class)

#         img_files = [fn for fn in os.listdir(data_dir) 
#                         if any(fn.endswith(ext) for ext in included_extensions)]

#         for img_file in img_files:
    
#             img_path = os.path.join(data_dir, img_file)
#             img = Image.open(img_path)
#             img = img.convert(mode='RGB')
#             x = transform(img)
#             x = torch.unsqueeze(x, dim=0)
            
#             features = resnet_model(x)

#             img_filename_data.append(img_path)
#             feature_vector_data.append(features.data[0].numpy())
#             img_class_data.append(cat2int[img_class])
        
#         print("{}: {} processed".format(out_dataset_name, img_class))

#     return img_filename_data, feature_vector_data, img_class_data