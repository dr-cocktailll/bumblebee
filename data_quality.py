import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, isnull, when
from nuscenes.nuscenes import NuScenes

from config import CONFIG

class DataQualityValidator:
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def validate_nuscenes_structure(self, data_path: str) -> Dict:
        required_files = [
            'v1.0-trainval/sample.json',
            'v1.0-trainval/scene.json',
            'v1.0-trainval/sample_data.json',
            'v1.0-trainval/sample_annotation.json',
            'v1.0-trainval/instance.json',
            'v1.0-trainval/category.json',
            'v1.0-trainval/attribute.json',
            'v1.0-trainval/visibility.json',
            'v1.0-trainval/ego_pose.json',
            'v1.0-trainval/calibrated_sensor.json',
            'v1.0-trainval/sensor.json',
            'v1.0-trainval/log.json',
            'v1.0-trainval/map.json'
        ]
        
        missing_files = []
        file_sizes = {}
        
        for required_file in required_files:
            full_path = os.path.join(data_path, required_file)
            if not os.path.exists(full_path):
                missing_files.append(required_file)
            else:
                file_sizes[required_file] = os.path.getsize(full_path)
        
        validation_result = {
            'is_valid': len(missing_files) == 0,
            'errors': missing_files,
            'summary': {
                'total_files_checked': len(required_files),
                'missing_files_count': len(missing_files),
                'file_sizes': file_sizes
            }
        }
        
        if validation_result['is_valid']:
            try:
                nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
                validation_result['summary']['total_scenes'] = len(nusc.scene)
                validation_result['summary']['total_samples'] = len(nusc.sample)
                validation_result['summary']['total_annotations'] = len(nusc.sample_annotation)
            except Exception as e:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Failed to load nuScenes: {str(e)}")
        
        return validation_result
    
    def validate_sensor_data_integrity(self, data_path: str) -> Dict:
        try:
            nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=False)
            
            sensor_stats = {
                'lidar': {'total': 0, 'missing': 0, 'corrupt': 0},
                'camera': {'total': 0, 'missing': 0, 'corrupt': 0},
                'radar': {'total': 0, 'missing': 0, 'corrupt': 0}
            }
            
            for sample_data in nusc.sample_data[:1000]:
                sensor_name = nusc.get('sensor', 
                    nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])['sensor_token'])['channel']
                
                data_file_path = nusc.get_sample_data_path(sample_data['token'])
                
                if 'LIDAR' in sensor_name:
                    sensor_type = 'lidar'
                elif 'CAM' in sensor_name:
                    sensor_type = 'camera'
                elif 'RADAR' in sensor_name:
                    sensor_type = 'radar'
                else:
                    continue
                
                sensor_stats[sensor_type]['total'] += 1
                
                if not os.path.exists(data_file_path):
                    sensor_stats[sensor_type]['missing'] += 1
                elif os.path.getsize(data_file_path) == 0:
                    sensor_stats[sensor_type]['corrupt'] += 1
            
            total_issues = sum(stats['missing'] + stats['corrupt'] 
                             for stats in sensor_stats.values())
            
            return {
                'is_valid': total_issues == 0,
                'sensor_stats': sensor_stats,
                'total_issues': total_issues
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'sensor_stats': {},
                'total_issues': -1
            }
    
    def validate_processed_data(self, s3_path: str) -> Dict:
        spark = SparkSession.builder \
            .appName("DataQualityValidation") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()
        
        try:
            df = spark.read.parquet(s3_path)
            
            total_rows = df.count()
            
            null_counts = {}
            for column in df.columns:
                null_count = df.filter(col(column).isNull() | 
                                     isnan(col(column)) | 
                                     (col(column) == "")).count()
                null_counts[column] = null_count
            
            schema_validation = self._validate_schema(df.schema)
            
            partition_stats = self._validate_partitions(s3_path)
            
            data_quality_score = self._calculate_quality_score(
                total_rows, null_counts, schema_validation, partition_stats)
            
            validation_result = {
                'is_valid': data_quality_score > 0.95,
                'quality_score': data_quality_score,
                'summary': {
                    'total_rows': total_rows,
                    'null_counts': null_counts,
                    'schema_valid': schema_validation['is_valid'],
                    'partition_stats': partition_stats
                }
            }
            
            if not validation_result['is_valid']:
                validation_result['errors'] = []
                
                if data_quality_score <= 0.95:
                    validation_result['errors'].append(f"Data quality score too low: {data_quality_score}")
                
                if not schema_validation['is_valid']:
                    validation_result['errors'].extend(schema_validation['errors'])
                
                high_null_columns = [col for col, count in null_counts.items() 
                                   if count / total_rows > 0.1]
                if high_null_columns:
                    validation_result['errors'].append(f"High null rates in columns: {high_null_columns}")
        
        except Exception as e:
            validation_result = {
                'is_valid': False,
                'error': str(e),
                'summary': {}
            }
        
        finally:
            spark.stop()
        
        return validation_result
    
    def _validate_schema(self, schema) -> Dict:
        required_columns = [
            'sample_token', 'scene_token', 'timestamp', 'lidar_points',
            'camera_features', 'radar_points', 'velocity_features', 'spatial_features'
        ]
        
        actual_columns = [field.name for field in schema.fields]
        missing_columns = [col for col in required_columns if col not in actual_columns]
        
        return {
            'is_valid': len(missing_columns) == 0,
            'errors': [f"Missing required column: {col}" for col in missing_columns],
            'required_columns': required_columns,
            'actual_columns': actual_columns
        }
    
    def _validate_partitions(self, s3_path: str) -> Dict:
        try:
            bucket = s3_path.replace('s3://', '').split('/')[0]
            prefix = '/'.join(s3_path.replace('s3://', '').split('/')[1:])
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                Delimiter='/'
            )
            
            partitions = []
            if 'CommonPrefixes' in response:
                partitions = [p['Prefix'] for p in response['CommonPrefixes']]
            
            total_size = 0
            file_count = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.parquet'):
                        total_size += obj['Size']
                        file_count += 1
            
            return {
                'partition_count': len(partitions),
                'total_size_bytes': total_size,
                'total_files': file_count,
                'avg_file_size_mb': (total_size / file_count / 1024 / 1024) if file_count > 0 else 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'partition_count': -1,
                'total_size_bytes': -1,
                'total_files': -1,
                'avg_file_size_mb': -1
            }
    
    def _calculate_quality_score(self, total_rows: int, null_counts: Dict, 
                               schema_validation: Dict, partition_stats: Dict) -> float:
        if total_rows == 0:
            return 0.0
        
        null_penalty = sum(count for count in null_counts.values()) / (total_rows * len(null_counts))
        
        schema_score = 1.0 if schema_validation['is_valid'] else 0.5
        
        partition_score = 1.0
        if partition_stats.get('avg_file_size_mb', 0) < 50 or partition_stats.get('avg_file_size_mb', 0) > 2000:
            partition_score = 0.8
        
        quality_score = schema_score * (1.0 - null_penalty) * partition_score
        
        return max(0.0, min(1.0, quality_score))
    
    def generate_quality_report(self, validation_results: List[Dict]) -> str:
        report_lines = ["# Data Quality Report", ""]
        
        for i, result in enumerate(validation_results):
            report_lines.append(f"## Validation {i+1}")
            report_lines.append(f"**Status:** {'PASSED' if result['is_valid'] else 'FAILED'}")
            
            if 'summary' in result:
                report_lines.append("### Summary")
                for key, value in result['summary'].items():
                    report_lines.append(f"- {key}: {value}")
            
            if not result['is_valid'] and 'errors' in result:
                report_lines.append("### Errors")
                for error in result['errors']:
                    report_lines.append(f"- {error}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)