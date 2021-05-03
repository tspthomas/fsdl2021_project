import os
import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms

from PIL import Image
from torchvision.models import resnet

from fsdl_lib.data import load_or_create_features

img_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
cat2int = {
    'buildings': 0,
    'forest': 1,
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5}
int2cat = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'}
included_extensions = ['jpg', 'jpeg', 'png']


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


def resnet50_feature_extractor(pretrained=False, model_dir=None, **kwargs):
    model = Resnet50Features(resnet.Bottleneck,
                             [3, 4, 6, 3], **kwargs)
    if pretrained:
        url = resnet.model_urls['resnet50']
        model.load_state_dict(model_zoo.load_url(url, model_dir=model_dir))
    model.eval()
    return model


def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def extract_features(src_path, dest_path, data_folder, model, transform):

    src = os.path.join(src_path, data_folder)
    features_data = load_or_create_features(dest_path, data_folder)
    
    for folder in os.listdir(src):
        folder_path = os.path.join(src, folder)
        folder_images = [fn for fn in os.listdir(folder_path)
                         if any(fn.endswith(ext) for ext in included_extensions)]

        for img_file in folder_images:
            img_path = os.path.join(folder_path, img_file)

            if features_data.path_exists(img_path):
                continue

            img = Image.open(img_path)
            img = img.convert(mode='RGB')
            x = transform(img)
            x = torch.unsqueeze(x, dim=0)
            feat = model(x)

            features_data.add(feat.data[0].numpy(),
                              cat2int[folder],
                              img_path)

        print("{}: {} processed".format(data_folder, folder))

    return features_data 
