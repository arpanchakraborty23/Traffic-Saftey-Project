from pathlib import Path
import src.constants as constants
from datetime import datetime
import os

class DataConfiguration:
    def __init__(self):
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        artifacts = constants.ARTIFACTS_FOLDER
        self.artifacts_path = os.path.join(artifacts,timestamp)

class DataIngestionConfig:
    def __init__(self,traning_config:DataConfiguration):
        self.data_ingestion_artifacts= os.path.join(
            traning_config.artifacts_path, constants.DATA_INGESTION_DIR
        )

        self.raw_data_dir= os.path.join(
            self.data_ingestion_artifacts,constants.RAW_DATA_DIR
        )

        self.feature_store_dir = os.path.join(
            self.data_ingestion_artifacts, constants.FEATURE_STORE_Dir
        )

        self.dataset_url = constants.DATASET_URL

class DataValidationConfig:
    def __init__(self,traning_config:DataConfiguration):
        self.data_validation_artifacts = os.path.join(
            traning_config.artifacts_path, constants.DATA_VALIDATION_DIR_NAME
        )

        self.status_file_path = os.path.join(
            self.data_validation_artifacts,constants.DATA_VALIDATION_STATUS_FILE
        )
        self.all_required_files = constants.DATA_VALIDATION_ALL_REQUIRED_FILES

class ModelTrainerConfig:
    def __init__(self,traning_config:DataConfiguration):
        self.model_train_dir = os.path.join(
            traning_config.artifacts_path,constants.MODEL_TRAIN_DIR
        )
        self.pre_trained_model_path = os.path.join(
            self.model_train_dir,constants.PRE_TRAIN_MODEL_DIR,constants.PRE_TRAIN_MODEL_FILE
        )
        self.outputs_path = os.path.join(
            self.model_train_dir,constants.TRAINED_MODEL_OUTPUT_PATH
        )
        self.model_path = os.path.join(
            self.model_train_dir,constants.TRAINED_MODEL_PATH
        )
        self.num_epochs = constants.TRAIN_EPOCHS
        self.batch_size = constants.BATCH_SIZE
        