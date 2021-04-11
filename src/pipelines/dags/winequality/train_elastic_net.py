import os
import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn

from datetime import timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


np.random.seed(33)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data_and_preprocess(**context):
    print("Loading Data")

    base_path = '/workspace/data/'
    csv_path = os.path.join(base_path, 'raw/winequality-red.csv')
    data = pd.read_csv(csv_path, sep=";")

    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]

    X_train_path = os.path.join(base_path,
                                'processed/X_train.npy')
    with open(X_train_path, 'wb') as f:
        np.save(f, X_train)

    y_train_path = os.path.join(base_path,
                                'processed/y_train.npy')
    with open(y_train_path, 'wb') as f:
        np.save(f, y_train)

    X_test_path = os.path.join(base_path,
                               'processed/X_test.npy')
    with open(X_test_path, 'wb') as f:
        np.save(f, X_test)

    y_test_path = os.path.join(base_path,
                               'processed/y_test.npy')
    with open(y_test_path, 'wb') as f:
        np.save(f, y_test)

    return "Data finished to load!" + str(len(data))


def train_model(**context):
    print("Train Model")
    base_path = '/workspace/data/'

    X_train_path = os.path.join(base_path,
                                'processed/X_train.npy')
    with open(X_train_path, 'rb') as f:
        X_train = np.load(f)

    y_train_path = os.path.join(base_path,
                                'processed/y_train.npy')
    with open(y_train_path, 'rb') as f:
        y_train = np.load(f)

    mlflow_run = mlflow.start_run()

    alpha = 0.5
    l1_ratio = 0.5
    lr_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr_model.fit(X_train, y_train)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    
    return lr_model, mlflow_run


def evaluate_model(**context):
    print("Evaluate Model")
    base_path = '/workspace/data/'

    X_test_path = os.path.join(base_path,
                            'processed/X_test.npy')
    with open(X_test_path, 'rb') as f:
        X_test = np.load(f)

    y_test_path = os.path.join(base_path,
                               'processed/y_test.npy')
    with open(y_test_path, 'rb') as f:
        y_test = np.load(f)

    # loads data from previous step
    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    lr_model = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    predicted_qualities = lr_model.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    return lr_model, active_run


def register_model(**context):
    # loads data from previous step
    task_instance = context['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    lr_model = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)
    
    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path='sklearn-model',
        registered_model_name='sklearn-elastic-net-model'
    )

    mlflow.end_run()

args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='wine_quality_train_elastic_net',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['winequality', 'training', 'elastic_net', 'scikit_learn']
) as dag:


    load_data_task = PythonOperator(
        task_id='load_data_and_preprocess', 
        python_callable=load_data_and_preprocess, 
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model', 
        python_callable=train_model, 
        dag=dag,
        provide_context=True
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model', 
        python_callable=evaluate_model, 
        dag=dag,
        provide_context=True
    )

    register_model_task = PythonOperator(
        task_id='register_model', 
        python_callable=register_model, 
        dag=dag
    )

    load_data_task >> train_model_task >> evaluate_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()