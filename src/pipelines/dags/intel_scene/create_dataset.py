import os
import numpy as np

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from dvc.api import make_checkpoint

from datetime import timedelta

from intel_scene.utils import extract_features_to_data, PROCESSED_DATA_DIR, Data

np.random.seed(33)

import pickle
def create_dataset():
    print("Creating Dataset")

    # train
    train = extract_features_to_data(dataset_name="train")

    with open(os.path.join(PROCESSED_DATA_DIR, 'train.pickle'), 'wb') as f:
        pickle.dump(train, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved Train Data")

    # test
    test = extract_features_to_data(dataset_name='test')    

    with open(os.path.join(PROCESSED_DATA_DIR, 'test.pickle'), 'wb') as f:
        pickle.dump(test, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Test Data")

    return

# def create_dataset():
#     print("Creating Dataset")

#     # train
#     _, X_train, y_train = create_np_arrays(in_data='raw', out_dataset_name="train")

#     with open(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), 'wb') as f:
#         np.save(f, X_train)

#     with open(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), 'wb') as f:
#         np.save(f, y_train)

#     print("Saved Train Data")


#     # test
#     _, X_test, y_test = create_np_arrays(in_data='raw', out_dataset_name="test")
    
#     with open(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), 'wb') as f:
#         np.save(f, X_test)

#     with open(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), 'wb') as f:
#         np.save(f, y_test)

#     print("Saved Test Data")

#     return

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='create_dataset',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'data prep', 'resnet50']
) as dag:

    create_dataset_task = PythonOperator(
        task_id='create_dataset', 
        python_callable=create_dataset, 
        dag=dag,
    )

    create_dataset_task 


if __name__ == "__main__":
    dag.cli()