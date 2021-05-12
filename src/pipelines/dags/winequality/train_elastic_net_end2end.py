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


def compute_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def save_numpy_array(dest_path, data, name):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    path = os.path.join(dest_path, name)
    with open(path, 'wb') as f:
        np.save(f, data)


def load_numpy_array(src_path, name):
    path = os.path.join(src_path, name)
    with open(path, 'rb') as f:
        data = np.load(f)
    return data


def load_data_and_preprocess(**kwargs):
    print("Loading Data")

    # paths
    src_path = os.path.join(kwargs['src'], kwargs['name'], kwargs['filename'])
    dest_path = os.path.join(kwargs['dest'], kwargs['name'])

    # load data
    data = pd.read_csv(src_path, sep=";")
    train, test = train_test_split(data, random_state=kwargs['random_state'])

    X_train = train.drop(["quality"], axis=1)
    X_test = test.drop(["quality"], axis=1)
    y_train = train[["quality"]]
    y_test = test[["quality"]]

    # save arrays to disk
    save_numpy_array(dest_path, X_train, 'X_train.npy')
    save_numpy_array(dest_path, y_train, 'y_train.npy')
    save_numpy_array(dest_path, X_test, 'X_test.npy')
    save_numpy_array(dest_path, y_test, 'y_test.npy')

    print("Data finished to load!")


def train_model(**kwargs):
    print("Train Model")

    # paths
    src_path = os.path.join(kwargs['dest'], kwargs['name'])

    # load training data
    X_train = load_numpy_array(src_path, kwargs['X_train_filename'])
    y_train = load_numpy_array(src_path, kwargs['y_train_filename'])

    # create or get existing experiment
    experiment = mlflow.get_experiment_by_name(kwargs['experiment_name'])

    if experiment:
        print("Experiment already existings. Using the same one.")
        experiment_id = experiment.experiment_id
    else:
        print("Experiment does not exist. Creating new one")
        experiment_id = mlflow.create_experiment(kwargs['experiment_name'])     

    mlflow_run = mlflow.start_run(experiment_id=experiment_id)
    mlflow.sklearn.autolog(log_models=False)

    alpha = 0.5
    l1_ratio = 0.5
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       random_state=kwargs['random_state'])
    model.fit(X_train, y_train)

    return model, mlflow_run


def eval_model(**kwargs):
    print("Evaluate Model")
    
    # paths
    src_path = os.path.join(kwargs['dest'], kwargs['name'])

    # load test data
    X_test = load_numpy_array(src_path, kwargs['X_test_filename'])
    y_test = load_numpy_array(src_path, kwargs['y_test_filename'])

    # loads data from previous step
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    lr_model = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    # evaluate trained model
    predicted_qualities = lr_model.predict(X_test)
    (rmse, mae, r2) = compute_metrics(y_test, predicted_qualities)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('r2', r2)
    mlflow.log_metric('mae', mae)

    return lr_model, active_run


def register_model(**kwargs):
    # loads data from previous step
    task_instance = kwargs['ti']
    task_instance_data = task_instance.xcom_pull(task_ids='train_model')
    lr_model = task_instance_data[0]
    active_run = task_instance_data[1]

    mlflow.start_run(active_run.info.run_id)

    mlflow.sklearn.log_model(
        sk_model=lr_model,
        artifact_path='model',
        registered_model_name='winequality_elastic_net'
    )

    mlflow.end_run()


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='wine_quality_train_elastic_net_end2end',
    default_args=args,
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['winequality', 'training', 'elastic_net', 'scikit_learn', 'end_to_end']
) as dag:

    dag_kwargs = {
        'src':  os.environ.get('RAW_DATA_DIR'),
        'dest':  os.environ.get('PROCESSED_DATA_DIR'),
        'name': 'winequality',
        'filename': 'winequality-red.csv',
        'random_state': 33,
        'X_train_filename': 'X_train.npy',
        'y_train_filename': 'y_train.npy',
        'X_test_filename': 'X_test.npy',
        'y_test_filename': 'y_test.npy',
        'experiment_name': 'winequality'
    }

    load_data_task = PythonOperator(
        task_id='load_data_and_preprocess',
        python_callable=load_data_and_preprocess,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    eval_model_task = PythonOperator(
        task_id='eval_model',
        python_callable=eval_model,
        op_kwargs=dag_kwargs,
        dag=dag
    )

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
        dag=dag
    )

    load_data_task >> train_model_task >> eval_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()
