import tensorflow as tf

from datetime import timedelta

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


def load_data():
    print("Loading Data")
    data = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    return "Size of the data: " + str(len(data))


def extract_features():
    print("Extract Features")


def train_model():
    print("Train Model")


def evaluate_model():
    print("Evaluate Model")


def register_model():
    print("Register Model")


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='training_keras_example',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    tags=['example', 'example2'],
    params={"example_key": "example_value"},
) as dag:


    load_data_task = PythonOperator(
        task_id='load_data', 
        python_callable=load_data, 
        dag=dag
    )

    extract_features_task = PythonOperator(
        task_id='extract_features', 
        python_callable=extract_features, 
        dag=dag
    )

    train_model_task = PythonOperator(
        task_id='train_model', 
        python_callable=train_model, 
        dag=dag
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model', 
        python_callable=evaluate_model, 
        dag=dag
    )

    register_model_task = PythonOperator(
        task_id='register_model', 
        python_callable=register_model, 
        dag=dag
    )

    load_data_task >> extract_features_task >> train_model_task >> evaluate_model_task >> register_model_task


if __name__ == "__main__":
    dag.cli()