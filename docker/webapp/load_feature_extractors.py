from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo

restnet_url = resnet.model_urls['resnet50']
_ = model_zoo.load_url(restnet_url)
