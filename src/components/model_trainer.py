from src.logger import logging as lg
from ultralytics import YOLO
from ultralytics import settings
from src.entity.artifacts_entity import DataIngestionArtifacts, ModelTrainerArtifacts
from src.configuration.traning_config import ModelTrainerConfig
from pathlib import Path
import dagshub
import mlflow
import os
import json

# Initialize DagsHub for MLflow tracking
dagshub.init(repo_owner='arpanchakraborty23', repo_name='Traffic-Saftey-Project', mlflow=True)

class ModelTrainer:
    def __init__(self, model_train_config: ModelTrainerConfig, ingestion_artifacts: DataIngestionArtifacts):
        self.model_train_config = model_train_config
        self.ingestion_artifacts = ingestion_artifacts

    @staticmethod
    def save_as_json(obj, file_path):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(obj, f, indent=4)
    
    def initiate_model_trainer(self):
        try:
            # Define paths
            data_yaml_path = os.path.join(self.ingestion_artifacts.unzip_data_path, 'data.yaml')
            pretrain_model = self.model_train_config.pre_trained_model_path
            output_result_dir = self.model_train_config.outputs_path

            # Load pre-trained model
            model = YOLO(pretrain_model)

            # Training parameters
            epochs = self.model_train_config.num_epochs
            batch = self.model_train_config.batch_size

            with mlflow.start_run():
                # Log parameters to MLflow
                mlflow.log_param('epochs', epochs)
                mlflow.log_param('batch_size', batch)

                # Start training
                lg.info('Starting model training...')
                results = model.train(
                    data=os.path.abspath(data_yaml_path),
                    epochs=epochs,
                    batch=batch,
                    imgsz=640,
                    project=output_result_dir
                )
                lg.info('Model training completed.')

                # Model validation
                lg.info('Starting model validation...')
                results = model.val()
                lg.info('Model validation completed.')

                score= json.dumps(results.results_dict)

                # Save training results as JSON
                self.save_as_json(file_path='score/score.json',obj=score)
                

                # Save the trained model
                os.makedirs(os.path.dirname(self.model_train_config.trained_model_path), exist_ok=True)
                model.save(self.model_train_config.trained_model_path)

                os.makedirs(os.path.dirname('final_model/trained_model.pt'), exist_ok=True)
                model.save('final_model/trained_model.pt')
                
                lg.info(f'Model saved at: {self.model_train_config.trained_model_path}')

                # Log the model 
                mlflow.log_artifact(self.model_train_config.trained_model_path, artifact_path='model')
        
                lg.info('Model logged to MLflow registry.')

            # Create and return artifacts
            model_trainer_artifacts = ModelTrainerArtifacts(
                model_path=self.model_train_config.trained_model_path,
                model=model
            )
            return model_trainer_artifacts

        except Exception as e:
            lg.error(f"Error during model training: {e}")
            raise e
