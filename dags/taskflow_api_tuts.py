from airflow.sdk import dag, task 
from datetime import datetime, timedelta

default_args = {
    'owner': 'atharv',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

@dag(dag_id='dag_with_taskflow', default_args=default_args)
def hello_world_etl():
    
    @task(multiple_outputs=True)
    def get_name():
        return {'first_name': 'Atharv', 'last_name': 'Verma'}
    

    @task 
    def get_age():
        return 21
    
    @task 
    def greet(first_name, last_name, age):
        print(f"Hello World! My name is {first_name} {last_name} and I am {age} years old!")

    name_dict = get_name()
    age = get_age()
    greet(first_name=name_dict['first_name'], last_name=name_dict['last_name'], age=age)

greet_dag = hello_world_etl()