from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import boto3
import json

from config import CONFIG
from spark_processor import SparkNuScenesProcessor
from data_quality import DataQualityValidator
from model_features import ModelFeatureGenerator

default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=15),
    'max_active_runs': 1
}

dag = DAG(
    'nuscenes_processing_pipeline',
    default_args=default_args,
    description='End-to-end nuScenes data processing pipeline',
    schedule_interval='0 2 * * *',
    catchup=False,
    tags=['nuscenes', 'autonomous-driving', '3d-detection']
)

def validate_raw_data(**context):
    validator = DataQualityValidator()
    
    validation_results = validator.validate_nuscenes_structure(CONFIG.raw_data_path)
    
    if not validation_results['is_valid']:
        raise ValueError(f"Data validation failed: {validation_results['errors']}")
    
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    print(f"Data validation passed: {validation_results['summary']}")

def check_s3_credentials(**context):
    try:
        s3_client = boto3.client('s3')
        s3_client.head_bucket(Bucket=CONFIG.s3_bucket)
        print(f"S3 bucket {CONFIG.s3_bucket} is accessible")
    except Exception as e:
        raise ValueError(f"S3 access failed: {str(e)}")

def process_scene_batch(**context):
    scene_batch = context['params'].get('scene_batch', 'all')
    
    processor = SparkNuScenesProcessor()
    
    if scene_batch == 'all':
        processor.run_full_pipeline()
    else:
        processor.run_batch_pipeline(scene_batch)
    
    context['task_instance'].xcom_push(key='processing_status', value='completed')

def generate_model_features(**context):
    feature_generator = ModelFeatureGenerator()
    
    input_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}"
    output_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-features"
    
    feature_generator.create_model_ready_dataset(input_path, output_path)
    
    context['task_instance'].xcom_push(key='feature_path', value=output_path)

def validate_output_quality(**context):
    validator = DataQualityValidator()
    
    output_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}"
    
    quality_results = validator.validate_processed_data(output_path)
    
    if not quality_results['is_valid']:
        raise ValueError(f"Output validation failed: {quality_results['errors']}")
    
    print(f"Output validation passed: {quality_results['summary']}")

def cleanup_temp_files(**context):
    s3_client = boto3.client('s3')
    
    temp_prefix = f"{CONFIG.s3_prefix}-temp"
    
    response = s3_client.list_objects_v2(
        Bucket=CONFIG.s3_bucket,
        Prefix=temp_prefix
    )
    
    if 'Contents' in response:
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        
        s3_client.delete_objects(
            Bucket=CONFIG.s3_bucket,
            Delete={'Objects': objects_to_delete}
        )
        
        print(f"Cleaned up {len(objects_to_delete)} temporary files")

def send_completion_notification(**context):
    processing_stats = {
        'dag_run_id': context['dag_run'].dag_id,
        'execution_date': str(context['execution_date']),
        'total_samples': context['task_instance'].xcom_pull(
            task_ids='validate_raw_data', key='validation_results')['summary']['total_samples'],
        'output_path': context['task_instance'].xcom_pull(
            task_ids='generate_model_features', key='feature_path')
    }
    
    print(f"Pipeline completed successfully: {json.dumps(processing_stats, indent=2)}")

validate_data_task = PythonOperator(
    task_id='validate_raw_data',
    python_callable=validate_raw_data,
    dag=dag
)

check_s3_task = PythonOperator(
    task_id='check_s3_credentials',
    python_callable=check_s3_credentials,
    dag=dag
)

process_scenes_task = PythonOperator(
    task_id='process_scene_batch',
    python_callable=process_scene_batch,
    dag=dag
)

generate_features_task = PythonOperator(
    task_id='generate_model_features',
    python_callable=generate_model_features,
    dag=dag
)

validate_output_task = PythonOperator(
    task_id='validate_output_quality',
    python_callable=validate_output_quality,
    dag=dag
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag
)

notify_task = PythonOperator(
    task_id='send_completion_notification',
    python_callable=send_completion_notification,
    dag=dag
)

[validate_data_task, check_s3_task] >> process_scenes_task
process_scenes_task >> generate_features_task
generate_features_task >> validate_output_task
validate_output_task >> cleanup_task
cleanup_task >> notify_task