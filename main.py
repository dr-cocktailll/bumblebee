#!/usr/bin/env python3

import argparse
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from spark_processor import SparkNuScenesProcessor
from data_quality import DataQualityValidator
from model_features import ModelFeatureGenerator, DatasetStatistics
from s3_utils import S3DataManager, ParquetOptimizer, DatasetVersionManager

def run_data_validation():
    print("Running data validation...")
    validator = DataQualityValidator()
    
    structure_results = validator.validate_nuscenes_structure(CONFIG.raw_data_path)
    print(f"Structure validation: {'PASSED' if structure_results['is_valid'] else 'FAILED'}")
    
    if not structure_results['is_valid']:
        print(f"Errors: {structure_results['errors']}")
        return False
    
    sensor_results = validator.validate_sensor_data_integrity(CONFIG.raw_data_path)
    print(f"Sensor validation: {'PASSED' if sensor_results['is_valid'] else 'FAILED'}")
    
    if not sensor_results['is_valid']:
        print(f"Sensor issues: {sensor_results.get('total_issues', 'Unknown')}")
    
    return structure_results['is_valid'] and sensor_results['is_valid']

def run_spark_processing():
    print("Starting Spark processing...")
    processor = SparkNuScenesProcessor()
    processor.run_full_pipeline()
    print("Spark processing completed")

def run_feature_generation():
    print("Generating model features...")
    
    input_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}"
    output_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-features"
    
    feature_generator = ModelFeatureGenerator()
    feature_generator.create_model_ready_dataset(input_path, output_path)
    
    train_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-train"
    val_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-val"
    
    split_stats = feature_generator.create_training_validation_split(
        output_path, train_path, val_path
    )
    
    print(f"Dataset split completed: {split_stats}")

def run_s3_optimization():
    print("Optimizing S3 storage...")
    
    s3_manager = S3DataManager()
    s3_manager.setup_bucket_structure()
    
    optimizer = ParquetOptimizer(s3_manager)
    
    input_prefix = f"{CONFIG.s3_prefix}/raw"
    output_prefix = f"{CONFIG.s3_prefix}/optimized"
    
    optimizer.optimize_parquet_files(input_prefix, output_prefix)
    print("S3 optimization completed")

def run_output_validation():
    print("Validating output data...")
    validator = DataQualityValidator()
    
    output_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}"
    
    validation_results = validator.validate_processed_data(output_path)
    print(f"Output validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}")
    
    if not validation_results['is_valid']:
        print(f"Errors: {validation_results.get('errors', [])}")
    
    return validation_results['is_valid']

def create_dataset_version():
    print("Creating dataset version...")
    
    s3_manager = S3DataManager()
    version_manager = DatasetVersionManager(s3_manager)
    
    dataset_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-features"
    
    version_id = version_manager.create_dataset_version(
        version_name=f"nuscenes_processed_{datetime.now().strftime('%Y%m%d')}",
        dataset_path=dataset_path,
        metadata={
            "processing_date": datetime.now().isoformat(),
            "lidar_sweeps": CONFIG.lidar_sweeps,
            "max_lidar_points": CONFIG.max_lidar_points,
            "classes": CONFIG.classes
        }
    )
    
    print(f"Created dataset version: {version_id}")
    return version_id

def generate_statistics():
    print("Generating dataset statistics...")
    
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("DatasetStatistics") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()
    
    stats_generator = DatasetStatistics(spark)
    
    dataset_path = f"s3://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}-features"
    stats = stats_generator.compute_dataset_statistics(dataset_path)
    
    print(f"Dataset Statistics:")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Unique Scenes: {stats['unique_scenes']}")
    print(f"  Locations: {[loc['location'] for loc in stats['locations']]}")
    
    spark.stop()
    return stats

def run_full_pipeline():
    print("=" * 60)
    print("NUSCENES DATA PIPELINE - FULL EXECUTION")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        if not run_data_validation():
            print("Data validation failed. Aborting pipeline.")
            return False
        
        run_spark_processing()
        
        run_feature_generation()
        
        if not run_output_validation():
            print("Output validation failed. Pipeline completed with warnings.")
        
        run_s3_optimization()
        
        version_id = create_dataset_version()
        
        stats = generate_statistics()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Start Time: {start_time}")
        print(f"End Time: {end_time}")
        print(f"Duration: {duration}")
        print(f"Dataset Version: {version_id}")
        print(f"Total Samples Processed: {stats.get('total_samples', 'N/A')}")
        print("Pipeline completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='nuScenes Data Processing Pipeline')
    
    parser.add_argument('--command', 
                       choices=['validate', 'process', 'features', 'optimize', 'full'],
                       default='full',
                       help='Command to execute')
    
    parser.add_argument('--raw-data-path', 
                       default=CONFIG.raw_data_path,
                       help='Path to raw nuScenes data')
    
    parser.add_argument('--s3-bucket', 
                       default=CONFIG.s3_bucket,
                       help='S3 bucket for processed data')
    
    args = parser.parse_args()
    
    CONFIG.raw_data_path = args.raw_data_path
    CONFIG.s3_bucket = args.s3_bucket
    
    if args.command == 'validate':
        success = run_data_validation()
    elif args.command == 'process':
        run_spark_processing()
        success = True
    elif args.command == 'features':
        run_feature_generation()
        success = True
    elif args.command == 'optimize':
        run_s3_optimization()
        success = True
    elif args.command == 'full':
        success = run_full_pipeline()
    else:
        print(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()