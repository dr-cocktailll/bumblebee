import boto3
import json
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from datetime import datetime
import logging

from config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3DataManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
    
    def setup_bucket_structure(self):
        try:
            self.s3_client.head_bucket(Bucket=CONFIG.s3_bucket)
            logger.info(f"Bucket {CONFIG.s3_bucket} already exists")
        except:
            self.s3_client.create_bucket(
                Bucket=CONFIG.s3_bucket,
                CreateBucketConfiguration={'LocationConstraint': 'us-west-2'}
            )
            logger.info(f"Created bucket {CONFIG.s3_bucket}")
        
        self._configure_bucket_lifecycle()
        self._configure_bucket_versioning()
    
    def _configure_bucket_lifecycle(self):
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'raw-data-transition',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': f'{CONFIG.s3_prefix}/raw/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': 'processed-data-transition',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': f'{CONFIG.s3_prefix}/processed/'},
                    'Transitions': [
                        {
                            'Days': 60,
                            'StorageClass': 'STANDARD_IA'
                        }
                    ]
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=CONFIG.s3_bucket,
            LifecycleConfiguration=lifecycle_config
        )
    
    def _configure_bucket_versioning(self):
        self.s3_client.put_bucket_versioning(
            Bucket=CONFIG.s3_bucket,
            VersioningConfiguration={'Status': 'Enabled'}
        )
    
    def upload_file(self, local_path: str, s3_key: str, 
                   metadata: Dict = None, storage_class: str = 'STANDARD'):
        extra_args = {
            'StorageClass': storage_class
        }
        
        if metadata:
            extra_args['Metadata'] = metadata
        
        self.s3_client.upload_file(
            local_path, CONFIG.s3_bucket, s3_key, ExtraArgs=extra_args
        )
        
        logger.info(f"Uploaded {local_path} to s3://{CONFIG.s3_bucket}/{s3_key}")
    
    def download_file(self, s3_key: str, local_path: str):
        self.s3_client.download_file(CONFIG.s3_bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{CONFIG.s3_bucket}/{s3_key} to {local_path}")
    
    def list_objects(self, prefix: str) -> List[Dict]:
        response = self.s3_client.list_objects_v2(
            Bucket=CONFIG.s3_bucket,
            Prefix=prefix
        )
        
        return response.get('Contents', [])
    
    def get_object_metadata(self, s3_key: str) -> Dict:
        response = self.s3_client.head_object(
            Bucket=CONFIG.s3_bucket,
            Key=s3_key
        )
        
        return {
            'size': response['ContentLength'],
            'last_modified': response['LastModified'],
            'storage_class': response.get('StorageClass', 'STANDARD'),
            'metadata': response.get('Metadata', {})
        }
    
    def create_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        return self.s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': CONFIG.s3_bucket, 'Key': s3_key},
            ExpiresIn=expiration
        )

class ParquetOptimizer:
    def __init__(self, s3_manager: S3DataManager):
        self.s3_manager = s3_manager
    
    def optimize_parquet_files(self, input_prefix: str, output_prefix: str):
        objects = self.s3_manager.list_objects(input_prefix)
        parquet_files = [obj for obj in objects if obj['Key'].endswith('.parquet')]
        
        file_groups = self._group_files_by_size(parquet_files)
        
        for group in file_groups:
            if self._should_merge_group(group):
                self._merge_parquet_files(group, output_prefix)
            elif self._should_split_group(group):
                self._split_parquet_files(group, output_prefix)
            else:
                self._copy_optimal_files(group, output_prefix)
    
    def _group_files_by_size(self, parquet_files: List[Dict]) -> List[List[Dict]]:
        small_files = [f for f in parquet_files if f['Size'] < 128 * 1024 * 1024]
        large_files = [f for f in parquet_files if f['Size'] > 2 * 1024 * 1024 * 1024]
        optimal_files = [f for f in parquet_files if 
                        128 * 1024 * 1024 <= f['Size'] <= 2 * 1024 * 1024 * 1024]
        
        groups = []
        
        if small_files:
            groups.append(small_files)
        
        for large_file in large_files:
            groups.append([large_file])
        
        for optimal_file in optimal_files:
            groups.append([optimal_file])
        
        return groups
    
    def _should_merge_group(self, group: List[Dict]) -> bool:
        total_size = sum(f['Size'] for f in group)
        return len(group) > 1 and total_size < 2 * 1024 * 1024 * 1024
    
    def _should_split_group(self, group: List[Dict]) -> bool:
        return len(group) == 1 and group[0]['Size'] > 2 * 1024 * 1024 * 1024
    
    def _merge_parquet_files(self, files: List[Dict], output_prefix: str):
        import pandas as pd
        
        dataframes = []
        
        for file_info in files:
            temp_path = f"/tmp/{os.path.basename(file_info['Key'])}"
            self.s3_manager.download_file(file_info['Key'], temp_path)
            
            df = pd.read_parquet(temp_path)
            dataframes.append(df)
            
            os.remove(temp_path)
        
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        output_key = f"{output_prefix}/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        temp_output = f"/tmp/merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        merged_df.to_parquet(
            temp_output, 
            compression=CONFIG.parquet_compression,
            row_group_size=CONFIG.parquet_row_group_size
        )
        
        self.s3_manager.upload_file(temp_output, output_key)
        os.remove(temp_output)
        
        logger.info(f"Merged {len(files)} files into {output_key}")
    
    def _split_parquet_files(self, files: List[Dict], output_prefix: str):
        import pandas as pd
        
        for file_info in files:
            temp_path = f"/tmp/{os.path.basename(file_info['Key'])}"
            self.s3_manager.download_file(file_info['Key'], temp_path)
            
            df = pd.read_parquet(temp_path)
            
            chunk_size = 1000000
            num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                output_key = f"{output_prefix}/split_{os.path.splitext(os.path.basename(file_info['Key']))[0]}_{i}.parquet"
                temp_output = f"/tmp/split_{i}.parquet"
                
                chunk_df.to_parquet(
                    temp_output,
                    compression=CONFIG.parquet_compression,
                    row_group_size=CONFIG.parquet_row_group_size
                )
                
                self.s3_manager.upload_file(temp_output, output_key)
                os.remove(temp_output)
            
            os.remove(temp_path)
            logger.info(f"Split {file_info['Key']} into {num_chunks} chunks")
    
    def _copy_optimal_files(self, files: List[Dict], output_prefix: str):
        for file_info in files:
            output_key = f"{output_prefix}/{os.path.basename(file_info['Key'])}"
            
            copy_source = {'Bucket': CONFIG.s3_bucket, 'Key': file_info['Key']}
            self.s3_manager.s3_resource.meta.client.copy(
                copy_source, CONFIG.s3_bucket, output_key
            )
            
            logger.info(f"Copied optimal file {file_info['Key']} to {output_key}")

class DatasetVersionManager:
    def __init__(self, s3_manager: S3DataManager):
        self.s3_manager = s3_manager
        self.metadata_key = f"{CONFIG.s3_prefix}/metadata/dataset_versions.json"
    
    def create_dataset_version(self, version_name: str, dataset_path: str, 
                             metadata: Dict = None) -> str:
        version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        version_info = {
            'version_id': version_id,
            'version_name': version_name,
            'dataset_path': dataset_path,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        versions = self._load_versions()
        versions[version_id] = version_info
        
        self._save_versions(versions)
        
        logger.info(f"Created dataset version {version_id}: {version_name}")
        return version_id
    
    def get_latest_version(self) -> Dict:
        versions = self._load_versions()
        
        if not versions:
            return None
        
        latest_key = max(versions.keys(), key=lambda k: versions[k]['created_at'])
        return versions[latest_key]
    
    def get_version(self, version_id: str) -> Dict:
        versions = self._load_versions()
        return versions.get(version_id)
    
    def list_versions(self) -> List[Dict]:
        versions = self._load_versions()
        return list(versions.values())
    
    def _load_versions(self) -> Dict:
        try:
            temp_path = "/tmp/dataset_versions.json"
            self.s3_manager.download_file(self.metadata_key, temp_path)
            
            with open(temp_path, 'r') as f:
                versions = json.load(f)
            
            os.remove(temp_path)
            return versions
        
        except:
            return {}
    
    def _save_versions(self, versions: Dict):
        temp_path = "/tmp/dataset_versions.json"
        
        with open(temp_path, 'w') as f:
            json.dump(versions, f, indent=2)
        
        self.s3_manager.upload_file(temp_path, self.metadata_key)
        os.remove(temp_path)