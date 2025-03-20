from src.logger import logging as lg
from src.configuration.traning_config import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifacts
import requests
import os
import zipfile

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """Download dataset from a URL."""
        response = requests.get(self.config.dataset_url)
        
        if response.status_code == 200:
            lg.info("Status code 200: Data retrieved successfully.")
        else:
            lg.error("Invalid URL or failed to retrieve data.")
            raise Exception("Failed to download data")

        # directory exists
        os.makedirs(os.path.dirname(self.config.raw_data_dir), exist_ok=True)

        # Save zip file properly
        with open(self.config.raw_data_dir, 'wb') as f:
            f.write(response.content)
        
        lg.info("Data Download Completed")

    def extract_data(self):
        """Extract the zip file to the feature store directory."""
        self.unzip_file_path = self.config.feature_store_dir
        os.makedirs(self.unzip_file_path, exist_ok=True)

        with zipfile.ZipFile(self.config.raw_data_dir, 'r') as z:
            z.extractall(self.unzip_file_path)

        lg.info("Data Extraction completed")

    def initiate_data_ingestion(self):
        """Coordinate data download and extraction."""
        try:
            self.download_data()
            self.extract_data()
            lg.info("Data Ingestion Completed")

            return DataIngestionArtifacts(
                zip_data_path=self.config.raw_data_dir,
                unzip_data_path=self.unzip_file_path
            )

        except Exception as e:
            lg.error(f"Data ingestion failed: {e}")
            raise e
