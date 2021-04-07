import os
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from datetime import timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from torchvision.models import resnet 

from PIL import Image

np.random.seed(33)

img_classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
cat2int = {'buildings': 0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
int2cat = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
included_extensions = ['jpg','jpeg','png']

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
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
    model.eval()
    return model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                normalize
                            ])

def create_np_arrays(dataset_name, data_dir):

    # load resnet model
    resnet_model = resnet50_feature_extractor(pretrained=True)

    # # loop through data_dir
    # # create np arrays of img_file, feature_vector, cat2int[img_class]
    img_filename_data = []
    feature_vector_data = []
    img_class_data = []
    for img_class in img_classes:
    
        raw_data_dir = os.path.join(data_dir, 
                    'raw/intel_image_scene/seg_{}/seg_{}'.format(dataset_name,dataset_name), img_class)

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


def create_dataset(**context):
    print("Creating Dataset")

    data_dir = '/workspace/data/' 

    # train
    img_filename_data, feature_vector_data, img_class_data = create_np_arrays(dataset_name="train", data_dir=data_dir)

    feature_vector_path = os.path.join(data_dir, 'processed/train_feature_vector.npy',)
    with open(feature_vector_path, 'wb') as f:
        np.save(f, feature_vector_data)

    img_filename_path = os.path.join(data_dir, 'processed/train_img_filename.npy')
    with open(img_filename_path, 'wb') as f:
        np.save(f, img_filename_data)

    img_class_path = os.path.join(data_dir, 'processed/train_img_class.npy')
    with open(img_class_path, 'wb') as f:
        np.save(f, img_class_data)

    print("Saved Train Data")

    # test
    img_filename_data, feature_vector_data, img_class_data = create_np_arrays(dataset_name="test", data_dir=data_dir)
    
    feature_vector_path = os.path.join(data_dir, 'processed/test_feature_vector.npy',)
    with open(feature_vector_path, 'wb') as f:
        np.save(f, feature_vector_data)

    img_filename_path = os.path.join(data_dir, 'processed/test_img_filename.npy')
    with open(img_filename_path, 'wb') as f:
        np.save(f, img_filename_data)

    img_class_path = os.path.join(data_dir, 'processed/test_img_class.npy')
    with open(img_class_path, 'wb') as f:
        np.save(f, img_class_data)

    print("Saved Test Data")

def train_model(**context):
    print("Train Model")
    data_dir = '/workspace/data/'
    model_path = '/workspace/src/ml/intel_scenes.pkl'

    X_train_path = os.path.join(data_dir,
                                'processed/train_feature_vector.npy')
    with open(X_train_path, 'rb') as f:
        X_train = np.load(f)

    y_train_path = os.path.join(data_dir,
                                'processed/train_img_class.npy')
    with open(y_train_path, 'rb') as f:
        y_train = np.load(f)

    with mlflow.start_run():    
        max_iter = 10000
        logmodel = LogisticRegression(max_iter=max_iter)
        logmodel.fit(X_train,y_train)

        mlflow.log_param("max_iter", max_iter)

    with open(model_path, 'wb') as f:
        pickle.dump(logmodel, f)

    # context['ti'].xcom_push(key='trained_model_lr', value=lr)

def eval_model(**context):
    print("Eval Model")
    data_dir = '/workspace/data/'
    model_path = '/workspace/src/ml/intel_scenes.pkl'

    X_test_path = os.path.join(data_dir,
                                'processed/test_feature_vector.npy')
    with open(X_test_path, 'rb') as f:
        X_test = np.load(f)

    y_test_path = os.path.join(data_dir,
                                'processed/test_img_class.npy')
    with open(y_test_path, 'rb') as f:
        y_test = np.load(f)

    # load model
    with open(model_path, 'rb') as f:
        logmodel = pickle.load(f)

    # check prediction report
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test, predictions))

    # context['ti'].xcom_push(key='trained_model_lr', value=lr)


# def register_model():
#     print("Register Model")


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='workflow_intel_scenes',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['scenes', 'intel']
) as dag:

    create_dataset_task = PythonOperator(
        task_id='load_data_and_preprocess', 
        python_callable=create_dataset, 
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model', 
        python_callable=train_model, 
        dag=dag,
        provide_context=True
    )

    eval_model_task = PythonOperator(
        task_id='eval_model', 
        python_callable=eval_model, 
        dag=dag,
        provide_context=True
    )

    create_dataset_task >> train_model_task >> eval_model_task

    # register_model_task = PythonOperator(
    #     task_id='register_model', 
    #     python_callable=register_model, 
    #     dag=dag
    # )

    # create_dataset_task >> train_model_task >> evaluate_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()