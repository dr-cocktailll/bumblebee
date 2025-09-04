import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class NuScenesConfig:
    raw_data_path: str = "/data/nuscenes"
    s3_bucket: str = "nuscenes-processed"
    s3_prefix: str = "v1.0-processed"
    
    lidar_sweeps: int = 10
    max_lidar_points: int = 150000
    temporal_window: int = 5
    
    spark_driver_memory: str = "8g"
    spark_executor_memory: str = "16g"
    spark_executor_cores: int = 4
    spark_max_executors: int = 20
    
    parquet_row_group_size: int = 50000
    parquet_compression: str = "snappy"
    
    camera_resolution: Tuple[int, int] = (1600, 900)
    radar_max_range: float = 250.0
    
    classes: List[str] = None
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = [
                'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
            ]

@dataclass
class SensorCalibration:
    translation: np.ndarray
    rotation: np.ndarray
    camera_intrinsic: Optional[np.ndarray] = None

@dataclass
class ProcessedSample:
    token: str
    scene_token: str
    timestamp: int
    ego_pose: Dict
    lidar_points: np.ndarray
    camera_features: Dict[str, np.ndarray]
    radar_points: np.ndarray
    annotations: List[Dict]
    velocity_features: np.ndarray
    spatial_features: Dict[str, np.ndarray]

CONFIG = NuScenesConfig()