from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator

from datetime import timedelta


args = {
    'owner': 'airflow',
}

with DAG(
    dag_id='intel_scenes_sync_feedback',
    default_args=args,
    schedule_interval='0 0 * * *',
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=5),
    tags=['intel_scenes', 'dataset']
) as dag:

    sync_dataset_task = BashOperator(
        task_id='sync_feedback',
        bash_command='./sync_feedback.sh',
        dag=dag,
    )

    sync_dataset_task

if __name__ == "__main__":
    dag.cli()
