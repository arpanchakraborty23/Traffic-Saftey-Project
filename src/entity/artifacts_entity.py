from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifacts:
    zip_data_path:Path
    unzip_data_path:Path

@dataclass
class DataValidationArtifact:
    status:bool

@dataclass
class ModelTrainerArtifacts:
    model_path:Path
    