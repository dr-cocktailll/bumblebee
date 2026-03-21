# bumblebee

End-to-end processing pipeline for the Panoptic nuScenes dataset. Transforms raw multi-modal sensor data (LiDAR, RADAR, camera) into model-ready features for 3D object detection, with object velocity as a core engineered feature. The pipeline is built on PySpark for distributed processing and Airflow for orchestration, writing partitioned Parquet files to S3.

## Pipeline

```
Raw nuScenes data
    |
    +-- validate (structure + sensor integrity)
    |
    +-- spark_processor (PySpark)
    |       LiDAR multi-sweep aggregation (ego-motion compensated)
    |       Camera ORB feature extraction (6 cameras)
    |       RADAR point processing (5 radars, incl. radial velocity)
    |       Object velocity estimation (temporal tracking)
    |       Panoptic segmentation labels (when available)
    |
    +-- model_features (feature engineering)
    |       Standardize + normalize sensor features
    |       3D detection targets (bbox, class, velocity)
    |       Train/val scene-level split
    |
    +-- validate output (quality scoring)
    |
    +-- S3 partitioned Parquet
            partitioned by location:
              singapore-onenorth/
              singapore-hollandvillage/
              singapore-queenstown/
              boston-seaport/
```

## Project Structure

```
├── config.py               Configuration and data classes
├── data_processors.py       Multi-modal sensor processing + panoptic labels
├── spark_processor.py       PySpark distributed processing
├── data_quality.py          Data validation and quality checks
├── model_features.py        Feature engineering for 3D detection
├── s3_utils.py              S3 storage and Parquet optimization
├── airflow_dag.py           Airflow DAG definition
├── main.py                  CLI entry point
└── requirements.txt         Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Needs:
- Spark 3.4+
- Airflow 2.7+
- AWS credentials for S3
- nuScenes dataset (v1.0-trainval with panoptic) at the configured `raw_data_path`

## Usage

Full pipeline:
```bash
python main.py --command full --raw-data-path /data/nuscenes --s3-bucket my-bucket
```

Individual stages:
```bash
python main.py --command validate
python main.py --command process
python main.py --command features
python main.py --command optimize
```

The Airflow DAG (`nuscenes_processing_pipeline`) runs daily at 2 AM UTC.

## Output

Parquet files partitioned by `location`, containing:

| Column | Description |
|--------|-------------|
| sample_token | nuScenes sample ID |
| lidar_features | Normalized multi-sweep LiDAR (150k pts x 4) |
| camera_features | ORB descriptors per camera (6 x 1024) |
| radar_features | Radar points with radial velocity |
| temporal_features | Aggregated velocity + spatial context |
| detection_targets | 3D boxes: [x,y,z,w,l,h,yaw] + class + velocity |
| semantic_labels | Per-point panoptic segmentation (when available) |
