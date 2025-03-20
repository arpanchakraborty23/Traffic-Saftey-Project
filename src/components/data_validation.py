import os
from src.logger import logging as lg
from src.configuration.traning_config import DataValidationConfig
from src.entity.artifacts_entity import DataValidationArtifact, DataIngestionArtifacts

class DataValidation:
    def __init__(self,
                 validation_config: DataValidationConfig,
                 ingestion_artifacts: DataIngestionArtifacts):
        self.validation_config = validation_config
        self.ingestion_artifacts = ingestion_artifacts

    def validate_all_file_exist(self):
        try:
            all_files = os.listdir(self.ingestion_artifacts.unzip_data_path)
            required_files = self.validation_config.all_required_files

            # Check if all required files are present
            missing_files = [file for file in required_files if file not in all_files]

            # Determine the validation status
            validation_status = len(missing_files) == 0

            # Save the status to file
            with open(self.validation_config.status_file_path, 'w') as f:
                if validation_status:
                    f.write(f"Validation status: {validation_status}")
                    lg.info("All files validated successfully.")
                else:
                    f.write(f"Validation status: {validation_status}\nMissing files: {', '.join(missing_files)}")
                    lg.error(f"Missing files: {', '.join(missing_files)}")

            return validation_status

        except Exception as e:
            lg.error(f"Error during file validation: {e}")
            raise e

    def initiate_data_val(self):
        try:
            # Ensure validation artifact directory exists
            validation_dir = os.path.dirname(self.validation_config.status_file_path)
            os.makedirs(validation_dir, exist_ok=True)

            # Perform file validation
            status = self.validate_all_file_exist()

            # Return validation artifact
            return DataValidationArtifact(
                status=status
            )

        except Exception as e:
            lg.error(f"Error during data validation initiation: {e}")
            raise e
