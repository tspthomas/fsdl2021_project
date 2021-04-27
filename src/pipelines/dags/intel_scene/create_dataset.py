import os
import pickle

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

from datetime import timedelta

from intel_scene.utils import extract_features_to_data, PROCESSED_DATA_DIR, Data


def create_dataset():
    print("Creating Dataset")

    data = extract_features_to_data()

    with open(os.path.join(PROCESSED_DATA_DIR, 'intel_image_scene', 'data.pickle'), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved Dataset")

    return


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_create_dataset',
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
