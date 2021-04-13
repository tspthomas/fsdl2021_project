import os

from PIL import Image

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision.models import resnet 

img_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
cat2int = {'buildings': 0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
int2cat = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
included_extensions = ['jpg','jpeg','png']

RAW_DATA_DIR = os.environ.get('RAW_DATA_DIR')
PROCESSED_DATA_DIR = os.environ.get('PROCESSED_DATA_DIR')

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

def create_np_arrays(dataset_name):

    # load resnet model
    resnet_model = resnet50_feature_extractor(pretrained=True)

    # # loop through data_dir
    # # create np arrays of img_file, feature_vector, cat2int[img_class]
    img_filename_data = []
    feature_vector_data = []
    img_class_data = []
    for img_class in img_classes:
    
        raw_data_dir = os.path.join(RAW_DATA_DIR, 
                    'intel_image_scene/seg_{}/seg_{}'.format(dataset_name,dataset_name), img_class)

        img_files = [fn for fn in os.listdir(raw_data_dir) 
                        if any(fn.endswith(ext) for ext in included_extensions)]

        for img_file in img_files:
    
            img_path = os.path.join(raw_data_dir, img_file)
            img = Image.open(img_path)
            img = img.convert(mode='RGB')
            x = transform(img)
            x = torch.unsqueeze(x, dim=0)
            
            features = resnet_model(x)

            img_filename_data.append(img_path)
            feature_vector_data.append(features.data[0].numpy())
            img_class_data.append(cat2int[img_class])
        
        print("{}: {} processed".format(dataset_name, img_class))

    return img_filename_data, feature_vector_data, img_class_data