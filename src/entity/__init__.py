from pathlib import Path
import src.constants as constants
from datetime import datetime
import os

class DataConfiguration:
    def __init__(self):
        timestamp = datetime().now()
        artifacts = constants.ARTIFACTS_FOLDER
        self.artifacts_path = os.path.join(artifacts,timestamp)

class DataIngestionConfig:
    def __init__(self,config:DataConfiguration):
        self.data_ingestion_artifacts= os.path.join(
            config.artifacts_path, constants.DATA_INGESTION_DIR
        )

        raw_data_dir= os.path.join(
            self.data_ingestion_artifacts,constants.RAW_DATA_DIR
        )

        feature_store_dir = os.path.join(
            self.data_ingestion_artifacts, constants.FEATURE_STORE_Dir
        )
