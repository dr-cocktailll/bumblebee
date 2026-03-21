import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, udf, explode, array, struct
from pyspark.sql.types import *
from typing import Dict, List, Tuple
import json

from config import CONFIG

NUSCENES_CATEGORY_REMAP = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.barrier': 'barrier',
}

CATEGORY_INDEX = {name: idx for idx, name in enumerate(CONFIG.classes)}


class ModelFeatureGenerator:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("ModelFeatureGeneration") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .getOrCreate()
    
    def create_model_ready_dataset(self, input_path: str, output_path: str):
        df = self.spark.read.parquet(input_path)
        
        model_df = self._create_standardized_features(df)
        model_df = self._create_temporal_features(model_df)
        model_df = self._create_detection_targets(model_df)
        model_df = self._normalize_features(model_df)
        
        self._write_model_dataset(model_df, output_path)
    
    def _create_standardized_features(self, df: DataFrame) -> DataFrame:
        @udf(returnType=ArrayType(DoubleType()))
        def standardize_lidar_points(points_list):
            if not points_list or len(points_list) == 0:
                return [0.0] * (CONFIG.max_lidar_points * 4)
            
            points = np.array(points_list)
            
            if points.shape[0] < CONFIG.max_lidar_points:
                padding = np.zeros((CONFIG.max_lidar_points - points.shape[0], points.shape[1]))
                points = np.vstack([points, padding])
            elif points.shape[0] > CONFIG.max_lidar_points:
                indices = np.random.choice(points.shape[0], CONFIG.max_lidar_points, replace=False)
                points = points[indices]
            
            return points.flatten().tolist()
        
        @udf(returnType=ArrayType(DoubleType()))
        def extract_camera_embedding(camera_features_map):
            if not camera_features_map:
                return [0.0] * 6144
            
            all_features = []
            camera_order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            
            for cam_name in camera_order:
                if cam_name in camera_features_map:
                    features = np.array(camera_features_map[cam_name])
                    if features.shape[0] > 0:
                        features_flat = features.flatten()[:1024]
                        if len(features_flat) < 1024:
                            padding = np.zeros(1024 - len(features_flat))
                            features_flat = np.concatenate([features_flat, padding])
                        all_features.extend(features_flat.tolist())
                    else:
                        all_features.extend([0.0] * 1024)
                else:
                    all_features.extend([0.0] * 1024)
            
            return all_features
        
        @udf(returnType=ArrayType(DoubleType()))
        def standardize_radar_points(radar_points_list):
            if not radar_points_list or len(radar_points_list) == 0:
                return [0.0] * 500
            
            radar_points = np.array(radar_points_list)
            
            max_radar_points = 100
            if radar_points.shape[0] < max_radar_points:
                padding = np.zeros((max_radar_points - radar_points.shape[0], radar_points.shape[1]))
                radar_points = np.vstack([radar_points, padding])
            elif radar_points.shape[0] > max_radar_points:
                indices = np.random.choice(radar_points.shape[0], max_radar_points, replace=False)
                radar_points = radar_points[indices]
            
            return radar_points.flatten().tolist()
        
        return df \
            .withColumn("lidar_features", standardize_lidar_points(col("lidar_points"))) \
            .withColumn("camera_embedding", extract_camera_embedding(col("camera_features"))) \
            .withColumn("radar_features", standardize_radar_points(col("radar_points")))
    
    def _create_temporal_features(self, df: DataFrame) -> DataFrame:
        @udf(returnType=ArrayType(DoubleType()))
        def create_temporal_context(velocity_features, spatial_features_map):
            temporal_features = []

            if velocity_features and len(velocity_features) >= 3:
                velocities = np.array(velocity_features).reshape(-1, 3)
                avg_velocity = np.mean(velocities, axis=0)
                max_velocity = float(np.max(np.linalg.norm(velocities, axis=1)))
                temporal_features.extend(avg_velocity.tolist())
                temporal_features.append(max_velocity)
            else:
                temporal_features.extend([0.0, 0.0, 0.0, 0.0])

            if spatial_features_map:
                if 'height_mean' in spatial_features_map:
                    temporal_features.extend(spatial_features_map['height_mean'])
                else:
                    temporal_features.append(0.0)

                if 'height_std' in spatial_features_map:
                    temporal_features.extend(spatial_features_map['height_std'])
                else:
                    temporal_features.append(0.0)
            else:
                temporal_features.extend([0.0, 0.0])

            return temporal_features

        return df.withColumn("temporal_features",
                           create_temporal_context(col("velocity_features"),
                                                 col("spatial_features")))
    
    def _create_detection_targets(self, df: DataFrame) -> DataFrame:
        @udf(returnType=ArrayType(StructType([
            StructField("class_id", IntegerType(), True),
            StructField("bbox_3d", ArrayType(DoubleType()), True),
            StructField("confidence", DoubleType(), True),
            StructField("velocity", ArrayType(DoubleType()), True)
        ])))
        def parse_annotations(annotations_json, velocity_features):
            if not annotations_json:
                return []

            try:
                annotations = json.loads(annotations_json)
                targets = []

                if velocity_features and len(velocity_features) >= 3:
                    velocities = np.array(velocity_features).reshape(-1, 3)
                else:
                    velocities = np.empty((0, 3))

                for i, ann in enumerate(annotations):
                    cat_name = ann['category_name']
                    det_class = NUSCENES_CATEGORY_REMAP.get(cat_name)
                    if det_class is None:
                        continue

                    class_id = CATEGORY_INDEX[det_class]

                    tx, ty, tz = ann['translation']
                    w, l, h = ann['size']

                    # yaw from quaternion [w, x, y, z]
                    qw, qx, qy, qz = ann['rotation']
                    yaw = float(np.arctan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy**2 + qz**2)
                    ))

                    bbox_3d = [tx, ty, tz, w, l, h, yaw]

                    vel = velocities[i].tolist() if i < len(velocities) else [0.0, 0.0, 0.0]

                    targets.append({
                        "class_id": class_id,
                        "bbox_3d": bbox_3d,
                        "confidence": 1.0,
                        "velocity": vel
                    })

                return targets
            except Exception:
                return []

        return df.withColumn("detection_targets",
                           parse_annotations(col("annotations"), col("velocity_features")))
    
    def _normalize_features(self, df: DataFrame) -> DataFrame:
        @udf(returnType=ArrayType(DoubleType()))
        def normalize_feature_vector(features):
            if not features or len(features) == 0:
                return features
            
            features_array = np.array(features)
            
            if np.std(features_array) > 0:
                normalized = (features_array - np.mean(features_array)) / np.std(features_array)
            else:
                normalized = features_array - np.mean(features_array)
            
            return normalized.tolist()
        
        return df \
            .withColumn("lidar_features_norm", normalize_feature_vector(col("lidar_features"))) \
            .withColumn("camera_embedding_norm", normalize_feature_vector(col("camera_embedding"))) \
            .withColumn("radar_features_norm", normalize_feature_vector(col("radar_features"))) \
            .withColumn("temporal_features_norm", normalize_feature_vector(col("temporal_features")))
    
    def _write_model_dataset(self, df: DataFrame, output_path: str):
        final_df = df.select(
            col("sample_token").alias("sample_id"),
            col("scene_token").alias("scene_id"),
            col("timestamp"),
            col("location"),
            col("lidar_features_norm").alias("lidar_features"),
            col("camera_embedding_norm").alias("camera_features"),
            col("radar_features_norm").alias("radar_features"),
            col("temporal_features_norm").alias("temporal_features"),
            col("detection_targets")
        )
        
        final_df.write \
            .mode("overwrite") \
            .option("compression", CONFIG.parquet_compression) \
            .partitionBy("location") \
            .parquet(output_path)
    
    def create_training_validation_split(self, input_path: str, train_path: str, 
                                       val_path: str, val_ratio: float = 0.2):
        df = self.spark.read.parquet(input_path)
        
        scenes = df.select("scene_id").distinct().collect()
        scene_list = [row["scene_id"] for row in scenes]
        
        np.random.shuffle(scene_list)
        val_size = int(len(scene_list) * val_ratio)
        
        val_scenes = scene_list[:val_size]
        train_scenes = scene_list[val_size:]
        
        train_df = df.filter(col("scene_id").isin(train_scenes))
        val_df = df.filter(col("scene_id").isin(val_scenes))
        
        train_df.write.mode("overwrite").parquet(train_path)
        val_df.write.mode("overwrite").parquet(val_path)
        
        return {
            "train_scenes": len(train_scenes),
            "val_scenes": len(val_scenes),
            "train_samples": train_df.count(),
            "val_samples": val_df.count()
        }

class DatasetStatistics:
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
    
    def compute_dataset_statistics(self, dataset_path: str) -> Dict:
        df = self.spark.read.parquet(dataset_path)
        
        stats = {
            "total_samples": df.count(),
            "unique_scenes": df.select("scene_id").distinct().count(),
            "locations": df.select("location").distinct().collect(),
            "feature_statistics": self._compute_feature_stats(df),
            "target_statistics": self._compute_target_stats(df)
        }
        
        return stats
    
    def _compute_feature_stats(self, df: DataFrame) -> Dict:
        feature_columns = ["lidar_features", "camera_features", "radar_features", "temporal_features"]
        
        stats = {}
        for col_name in feature_columns:
            sample_features = df.select(col_name).limit(1000).collect()
            
            if sample_features:
                feature_lengths = [len(row[col_name]) for row in sample_features if row[col_name]]
                
                stats[col_name] = {
                    "avg_length": np.mean(feature_lengths) if feature_lengths else 0,
                    "min_length": np.min(feature_lengths) if feature_lengths else 0,
                    "max_length": np.max(feature_lengths) if feature_lengths else 0
                }
        
        return stats
    
    def _compute_target_stats(self, df: DataFrame) -> Dict:
        targets_sample = df.select("detection_targets").limit(1000).collect()
        
        class_counts = {}
        total_objects = 0
        
        for row in targets_sample:
            if row["detection_targets"]:
                for target in row["detection_targets"]:
                    class_id = target["class_id"]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_objects += 1
        
        return {
            "total_objects": total_objects,
            "unique_classes": len(class_counts),
            "class_distribution": class_counts,
            "avg_objects_per_sample": total_objects / len(targets_sample) if targets_sample else 0
        }