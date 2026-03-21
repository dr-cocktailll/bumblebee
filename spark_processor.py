import logging
import json
from typing import Dict, List
import numpy as np
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType,
    ArrayType, DoubleType, MapType
)
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from config import CONFIG
from data_processors import (
    LidarProcessor, CameraProcessor, RadarProcessor,
    VelocityEstimator, SpatialFeatureExtractor, PanopticLabelLoader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkNuScenesProcessor:
    def __init__(self):
        self.spark = self._create_spark_session()
        self.nusc = None

    def _create_spark_session(self) -> SparkSession:
        return SparkSession.builder \
            .appName("NuScenesProcessor") \
            .config("spark.driver.memory", CONFIG.spark_driver_memory) \
            .config("spark.executor.memory", CONFIG.spark_executor_memory) \
            .config("spark.executor.cores", str(CONFIG.spark_executor_cores)) \
            .config("spark.dynamicAllocation.enabled", "true") \
            .config("spark.dynamicAllocation.maxExecutors",
                    str(CONFIG.spark_max_executors)) \
            .config("spark.hadoop.fs.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.sql.parquet.compression.codec",
                    CONFIG.parquet_compression) \
            .config("spark.serializer",
                    "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

    def _load_nuscenes(self):
        if self.nusc is None:
            logger.info(f"Loading nuScenes from {CONFIG.raw_data_path}")
            self.nusc = NuScenes(
                version='v1.0-trainval',
                dataroot=CONFIG.raw_data_path,
                verbose=False
            )
            logger.info(f"Loaded {len(self.nusc.scene)} scenes, "
                        f"{len(self.nusc.sample)} samples")

    def run_full_pipeline(self):
        self._load_nuscenes()
        scene_tokens = [s['token'] for s in self.nusc.scene]
        logger.info(f"Processing all {len(scene_tokens)} scenes")

        processed = self._process_scenes(scene_tokens)
        self._write_partitioned_parquet(processed)
        logger.info(f"Pipeline complete: {len(processed)} samples written")

    def run_batch_pipeline(self, scene_tokens: List[str]):
        self._load_nuscenes()
        all_scene_tokens = {s['token'] for s in self.nusc.scene}
        valid_tokens = [t for t in scene_tokens if t in all_scene_tokens]

        if len(valid_tokens) != len(scene_tokens):
            logger.warning(f"Skipping {len(scene_tokens) - len(valid_tokens)} "
                           "invalid scene tokens")

        processed = self._process_scenes(valid_tokens)
        self._write_partitioned_parquet(processed)
        logger.info(f"Batch complete: {len(processed)} samples written")

    def _process_scenes(self, scene_tokens: List[str]) -> List[Dict]:
        lidar_proc = LidarProcessor(self.nusc)
        camera_proc = CameraProcessor(self.nusc)
        radar_proc = RadarProcessor(self.nusc)
        velocity_est = VelocityEstimator(CONFIG.temporal_window)
        panoptic_loader = PanopticLabelLoader(self.nusc)

        all_samples = []
        for scene_token in tqdm(scene_tokens, desc="Scenes"):
            scene = self.nusc.get('scene', scene_token)
            scene_samples = self._process_scene(
                scene, lidar_proc, camera_proc, radar_proc,
                velocity_est, panoptic_loader
            )
            all_samples.extend(scene_samples)

        return all_samples

    def _process_scene(self, scene: Dict, lidar_proc, camera_proc,
                       radar_proc, velocity_est, panoptic_loader) -> List[Dict]:
        log = self.nusc.get('log', scene['log_token'])
        location = log['location']

        samples = []
        sample_token = scene['first_sample_token']
        while sample_token != '':
            sample = self.nusc.get('sample', sample_token)
            try:
                row = self._process_sample(
                    sample, location, lidar_proc, camera_proc,
                    radar_proc, velocity_est, panoptic_loader
                )
                samples.append(row)
            except Exception as e:
                logger.error(f"Failed sample {sample_token}: {e}")
            sample_token = sample['next']

        return samples

    def _process_sample(self, sample: Dict, location: str,
                        lidar_proc, camera_proc, radar_proc,
                        velocity_est, panoptic_loader) -> Dict:
        token = sample['token']

        lidar_sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])

        lidar_points = lidar_proc.load_point_cloud_multisweep(
            token, CONFIG.lidar_sweeps
        )
        camera_features = camera_proc.extract_camera_features(token)
        radar_points = radar_proc.process_radar_data(token)

        # panoptic per-point semantic labels (keyframe only)
        lidar_token = sample['data']['LIDAR_TOP']
        semantic_labels = panoptic_loader.load_labels(lidar_token)

        annotations = [self.nusc.get('sample_annotation', t)
                       for t in sample['anns']]

        velocity_features = velocity_est.calculate_object_velocity(
            annotations, token, sample['timestamp']
        )

        spatial_features = {}
        if lidar_points.shape[0] > 0:
            spatial_features = SpatialFeatureExtractor.extract_height_features(
                lidar_points
            )

        ann_records = []
        for ann in annotations:
            ann_records.append({
                'token': ann['token'],
                'category_name': ann['category_name'],
                'translation': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'num_lidar_pts': ann['num_lidar_pts'],
                'num_radar_pts': ann['num_radar_pts'],
            })

        result = {
            'sample_token': token,
            'scene_token': sample['scene_token'],
            'timestamp': sample['timestamp'],
            'location': location,
            'ego_translation': json.dumps(ego_pose['translation']),
            'ego_rotation': json.dumps(ego_pose['rotation']),
            'lidar_points': lidar_points.flatten().tolist(),
            'lidar_num_points': int(lidar_points.shape[0]),
            'camera_features': {k: v.flatten().tolist()
                                for k, v in camera_features.items()},
            'radar_points': radar_points.flatten().tolist(),
            'radar_num_points': int(radar_points.shape[0]),
            'annotations': json.dumps(ann_records),
            'num_annotations': len(annotations),
            'velocity_features': velocity_features.flatten().tolist(),
            'spatial_features': {k: v.tolist()
                                 for k, v in spatial_features.items()},
        }

        if semantic_labels is not None:
            result['semantic_labels'] = semantic_labels.tolist()
        else:
            result['semantic_labels'] = []

        return result

    def _build_schema(self) -> StructType:
        return StructType([
            StructField('sample_token', StringType(), False),
            StructField('scene_token', StringType(), False),
            StructField('timestamp', LongType(), False),
            StructField('location', StringType(), False),
            StructField('ego_translation', StringType(), True),
            StructField('ego_rotation', StringType(), True),
            StructField('lidar_points', ArrayType(DoubleType()), True),
            StructField('lidar_num_points', IntegerType(), True),
            StructField('camera_features',
                        MapType(StringType(), ArrayType(DoubleType())), True),
            StructField('radar_points', ArrayType(DoubleType()), True),
            StructField('radar_num_points', IntegerType(), True),
            StructField('annotations', StringType(), True),
            StructField('num_annotations', IntegerType(), True),
            StructField('velocity_features', ArrayType(DoubleType()), True),
            StructField('spatial_features',
                        MapType(StringType(), ArrayType(DoubleType())), True),
            StructField('semantic_labels', ArrayType(IntegerType()), True),
        ])

    def _write_partitioned_parquet(self, processed_samples: List[Dict]):
        if not processed_samples:
            logger.warning("No samples to write")
            return

        schema = self._build_schema()
        rows = [Row(**s) for s in processed_samples]
        df = self.spark.createDataFrame(rows, schema=schema)

        output_path = f"s3a://{CONFIG.s3_bucket}/{CONFIG.s3_prefix}"

        df.repartition("location") \
            .write \
            .mode("overwrite") \
            .partitionBy("location") \
            .option("compression", CONFIG.parquet_compression) \
            .parquet(output_path)

        row_count = df.count()
        partitions = df.select("location").distinct().count()
        logger.info(f"Wrote {row_count} samples across {partitions} "
                    f"location partitions to {output_path}")

    def shutdown(self):
        if self.spark:
            self.spark.stop()
