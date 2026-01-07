from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

defaults_args = {
    'owner': 'atharv',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

def greet(ti):
    first_name = ti.xcom_pull(task_ids='get_name', key='first_name')
    last_name = ti.xcom_pull(task_ids='get_name', key='last_name')
    age = ti.xcom_pull(task_ids='get_age', key='age')
    print(f'Hello World! My name is {first_name} {last_name}, and I am {age} years old!')

def get_name(ti):
    ti.xcom_push(key='first_name', value='Atharv')
    ti.xcom_push(key='last_name', value='Verma')

def get_age(ti):
    ti.xcom_push(key='age', value=21)

with DAG(
    default_args=defaults_args,
    dag_id='our_first_day',
    description='Our first dag'
) as dag:
    task1 = PythonOperator(
        task_id='greet',
        python_callable=greet
    )

    task2 = PythonOperator(
        task_id='get_name',
        python_callable=get_name
    )

    task3 = PythonOperator(
        task_id='get_age',
        python_callable=get_age
    )
    [task2 >> task3] >> task1