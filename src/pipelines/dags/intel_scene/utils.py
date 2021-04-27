import os
import torch

from PIL import Image
from fsdl_lib import feature_extraction as fe


included_extensions = ['jpg', 'jpeg', 'png']
FEEDBACK_DATA_DIR = os.environ.get('FEEDBACK_DATA_DIR')
RAW_DATA_DIR = os.environ.get('RAW_DATA_DIR')
PROCESSED_DATA_DIR = os.environ.get('PROCESSED_DATA_DIR')


class Data():
    def __init__(self):
        self.seen_features = {}
        self.X = []
        self.y = []


def extract_features_to_data(preload_dict=False):
    '''
    preload_dict (bool): preload or not

    returns: (Data)
    '''
    # set model_dir due to airflow container permissions
    model_dir = '/opt/airflow/.cache/torch'
    resnet_model = fe.resnet50_feature_extractor(pretrained=True,
                                                 model_dir=model_dir)
    transform = fe.get_transform()

    data = Data()

    if preload_dict:
        with open(os.path.join(PROCESSED_DATA_DIR, 'intel_image_scene', 'data.pickle'), 'rb') as f:
            data = pickle.load(f)

    for img_class in fe.img_classes:

        data_dir = os.path.join(RAW_DATA_DIR,
                                'intel_image_scene',
                                img_class)

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
                data.y.append(fe.cat2int[img_class])

        print("{} processed".format(img_class))

    return data
