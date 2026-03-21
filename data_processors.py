import os
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, List, Tuple, Optional
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import transform_matrix
import json

class CoordinateTransformer:
    def __init__(self):
        pass
    
    @staticmethod
    def ego_to_global(points: np.ndarray, ego_pose: Dict) -> np.ndarray:
        translation = np.array(ego_pose['translation'])
        rotation = Rotation.from_quat(ego_pose['rotation']).as_matrix()
        
        transformed = (rotation @ points.T).T + translation
        return transformed
    
    @staticmethod
    def sensor_to_ego(points: np.ndarray, calibration: Dict) -> np.ndarray:
        translation = np.array(calibration['translation'])
        rotation = Rotation.from_quat(calibration['rotation']).as_matrix()
        
        transformed = (rotation @ points.T).T + translation
        return transformed
    
    @staticmethod
    def create_transformation_matrix(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform

class LidarProcessor:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
    
    def load_point_cloud_multisweep(self, sample_token: str, num_sweeps: int = 10) -> np.ndarray:
        sample = self.nusc.get('sample', sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        
        all_points = []
        current_token = lidar_token
        
        for i in range(num_sweeps):
            if current_token == '':
                break
                
            lidar_data = self.nusc.get('sample_data', current_token)
            lidar_path = self.nusc.get_sample_data_path(current_token)
            
            pc = LidarPointCloud.from_file(lidar_path)
            points = pc.points.T[:, :3]
            
            if i > 0:
                ego_pose_current = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
                ego_pose_ref = self.nusc.get('ego_pose', 
                    self.nusc.get('sample_data', lidar_token)['ego_pose_token'])
                
                points = self._transform_to_reference_frame(points, ego_pose_current, ego_pose_ref)
            
            timestamps = np.full((points.shape[0], 1), i * 0.05)
            points_with_time = np.hstack([points, timestamps])
            all_points.append(points_with_time)
            
            current_token = lidar_data['prev']
        
        return np.vstack(all_points) if all_points else np.empty((0, 4))
    
    def _transform_to_reference_frame(self, points: np.ndarray, 
                                    current_pose: Dict, ref_pose: Dict) -> np.ndarray:
        current_transform = self._pose_to_transform_matrix(current_pose)
        ref_transform = self._pose_to_transform_matrix(ref_pose)
        
        relative_transform = np.linalg.inv(ref_transform) @ current_transform
        
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (relative_transform @ points_homogeneous.T).T
        
        return transformed_points[:, :3]
    
    def _pose_to_transform_matrix(self, pose: Dict) -> np.ndarray:
        translation = np.array(pose['translation'])
        rotation = Rotation.from_quat(pose['rotation']).as_matrix()
        return CoordinateTransformer.create_transformation_matrix(translation, rotation)

class CameraProcessor:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    def extract_camera_features(self, sample_token: str) -> Dict[str, np.ndarray]:
        sample = self.nusc.get('sample', sample_token)
        features = {}
        
        for cam_name in self.camera_names:
            cam_token = sample['data'][cam_name]
            cam_path = self.nusc.get_sample_data_path(cam_token)
            
            image = cv2.imread(cam_path)
            image = cv2.resize(image, (800, 450))
            
            features[cam_name] = self._extract_visual_features(image)
        
        return features
    
    def _extract_visual_features(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.zeros((1000, 32))
        
        if descriptors.shape[0] < 1000:
            padding = np.zeros((1000 - descriptors.shape[0], 32))
            descriptors = np.vstack([descriptors, padding])
        
        return descriptors[:1000]
    
    def project_lidar_to_camera(self, lidar_points: np.ndarray, 
                               camera_token: str) -> np.ndarray:
        cam_data = self.nusc.get('sample_data', camera_token)
        cam_calibration = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        
        camera_intrinsic = np.array(cam_calibration['camera_intrinsic'])
        
        lidar_to_cam = CoordinateTransformer.sensor_to_ego(
            lidar_points, cam_calibration)
        
        projected_points = (camera_intrinsic @ lidar_to_cam[:, :3].T).T
        projected_points = projected_points / projected_points[:, 2:3]
        
        return projected_points[:, :2]

class RadarProcessor:
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self.radar_names = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                           'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
    
    def process_radar_data(self, sample_token: str) -> np.ndarray:
        sample = self.nusc.get('sample', sample_token)
        all_radar_points = []
        
        for radar_name in self.radar_names:
            radar_token = sample['data'][radar_name]
            radar_path = self.nusc.get_sample_data_path(radar_token)
            
            radar_pc = RadarPointCloud.from_file(radar_path)
            points = radar_pc.points.T
            
            radar_data = self.nusc.get('sample_data', radar_token)
            calibration = self.nusc.get('calibrated_sensor', 
                                      radar_data['calibrated_sensor_token'])
            
            ego_points = CoordinateTransformer.sensor_to_ego(
                points[:, :3], calibration)
            
            velocity = points[:, 8:10] if points.shape[1] > 8 else np.zeros((points.shape[0], 2))
            
            radar_features = np.hstack([ego_points, velocity])
            all_radar_points.append(radar_features)
        
        return np.vstack(all_radar_points) if all_radar_points else np.empty((0, 5))

class VelocityEstimator:
    def __init__(self, temporal_window: int = 5):
        self.temporal_window = temporal_window
        self.previous_samples = {}
    
    def calculate_object_velocity(self, current_annotations: List[Dict],
                                sample_token: str, timestamp: int) -> np.ndarray:
        velocities = []
        
        for ann in current_annotations:
            instance_token = ann.get('instance_token', '')
            current_center = np.array(ann['translation'])
            
            if instance_token in self.previous_samples:
                prev_data = self.previous_samples[instance_token]
                
                if len(prev_data) >= 2:
                    velocity = self._estimate_velocity_kalman(prev_data, current_center, timestamp)
                else:
                    velocity = self._estimate_velocity_simple(prev_data[-1], current_center, timestamp)
            else:
                velocity = np.array([0.0, 0.0, 0.0])
            
            velocities.append(velocity)
            
            self._update_history(instance_token, current_center, timestamp)
        
        return np.array(velocities) if velocities else np.empty((0, 3))
    
    def _estimate_velocity_simple(self, prev_data: Dict, 
                                current_center: np.ndarray, timestamp: int) -> np.ndarray:
        dt = (timestamp - prev_data['timestamp']) / 1e6
        if dt <= 0:
            return np.array([0.0, 0.0, 0.0])
        
        velocity = (current_center - prev_data['center']) / dt
        return velocity
    
    def _estimate_velocity_kalman(self, history: List[Dict], 
                                current_center: np.ndarray, timestamp: int) -> np.ndarray:
        if len(history) < 3:
            return self._estimate_velocity_simple(history[-1], current_center, timestamp)
        
        positions = np.array([h['center'] for h in history[-3:]] + [current_center])
        times = np.array([h['timestamp'] for h in history[-3:]] + [timestamp]) / 1e6
        
        dt = np.diff(times)
        velocities = np.diff(positions, axis=0) / dt.reshape(-1, 1)
        
        return np.mean(velocities[-2:], axis=0)
    
    def _update_history(self, instance_token: str, center: np.ndarray, timestamp: int):
        if instance_token not in self.previous_samples:
            self.previous_samples[instance_token] = []
        
        self.previous_samples[instance_token].append({
            'center': center,
            'timestamp': timestamp
        })
        
        if len(self.previous_samples[instance_token]) > self.temporal_window:
            self.previous_samples[instance_token].pop(0)

class SpatialFeatureExtractor:
    @staticmethod
    def extract_density_features(lidar_points: np.ndarray, 
                               grid_size: Tuple[int, int, int] = (200, 200, 40)) -> np.ndarray:
        x_min, y_min, z_min = lidar_points.min(axis=0)[:3]
        x_max, y_max, z_max = lidar_points.max(axis=0)[:3]
        
        x_bins = np.linspace(x_min, x_max, grid_size[0] + 1)
        y_bins = np.linspace(y_min, y_max, grid_size[1] + 1)
        z_bins = np.linspace(z_min, z_max, grid_size[2] + 1)
        
        density_grid = np.histogramdd(lidar_points[:, :3], bins=[x_bins, y_bins, z_bins])[0]
        
        return density_grid.flatten()
    
    @staticmethod
    def extract_height_features(lidar_points: np.ndarray) -> Dict[str, np.ndarray]:
        z_values = lidar_points[:, 2]

        features = {
            'height_mean': np.array([np.mean(z_values)]),
            'height_std': np.array([np.std(z_values)]),
            'height_min': np.array([np.min(z_values)]),
            'height_max': np.array([np.max(z_values)]),
            'height_range': np.array([np.max(z_values) - np.min(z_values)])
        }

        return features


class PanopticLabelLoader:
    """Loads per-point semantic segmentation labels from Panoptic nuScenes."""

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        self._has_lidarseg = self._check_lidarseg()

    def _check_lidarseg(self) -> bool:
        lidarseg_path = os.path.join(
            self.nusc.dataroot, self.nusc.version, 'lidarseg.json'
        )
        return os.path.exists(lidarseg_path)

    def load_labels(self, lidar_token: str) -> Optional[np.ndarray]:
        if not self._has_lidarseg:
            return None
        try:
            lidarseg_record = self.nusc.get('lidarseg', lidar_token)
            labels_path = os.path.join(
                self.nusc.dataroot, lidarseg_record['filename']
            )
            return np.fromfile(labels_path, dtype=np.uint8)
        except Exception:
            return None