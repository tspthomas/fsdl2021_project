import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision.models import resnet

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
