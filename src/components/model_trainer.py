from src.logger import logging as lg
from ultralytics import YOLO
from ultralytics import settings
from src.entity.artifacts_entity import DataIngestionArtifacts,ModelTrainerArtifacts
from src.configuration.traning_config import ModelTrainerConfig
from pathlib import Path
import dagshub
import mlflow
import os

dagshub.init(repo_owner='arpanchakraborty23', repo_name='Traffic-Saftey-Project', mlflow=True)


class ModelTrainer:
    def __init__(self, model_train_config:ModelTrainerConfig, ingestion_artifacts:DataIngestionArtifacts):
        self.model_train_config = model_train_config
        self.ingestion_artifacts = ingestion_artifacts
    

    def initiate_model_trainer(self):
        try:
            data_yaml_path = self.ingestion_artifacts.unzip_data_path+'/data.yaml'
            pretrain_model = self.model_train_config.pre_trained_model_path
            output_result_dir = self.model_train_config.outputs_path

            # load model
            model = YOLO(pretrain_model)

            # params
            epochs = self.model_train_config.num_epochs
            batch = self.model_train_config.batch_size

            with mlflow.start_run():

                mlflow.log_param('epochs',epochs)
                mlflow.log_param('batch size',batch)

                # start Traning
                lg.info('start Traning ...')
                results = model.train(
                    data= os.path.abspath(data_yaml_path),
                    epochs = epochs,
                    batch = batch,
                    imgsz= 640,
                    project = output_result_dir
                )
                lg.info('Model tranin competed')

                # Access specific metrics
                mlflow.log_metric("Class indices with average precision:", results.ap_class_index)
                mlflow.log_metric("Average precision:", results.box.ap) 
                mlflow.log_metric("Mean average precision at IoU=0.50:", results.box.map50)
                mlflow.log_metric("Mean average precision at IoU=0.75:", results.box.map75)

                # model validation
                results= model.val()


                lg.info('model eval completed')
                # save model
                os.makedirs(os.path.dirname(self.model_train_config.trained_model_path),exist_ok=True)
                model.save(self.model_train_config.trained_model_path)
                mlflow.log_artifact("best_model.pt")

            lg.info(f'Model save at: {self.model_train_config.trained_model_path}')

            model_trainer_artifacts = ModelTrainerArtifacts(
                model_path= self.model_train_config.trained_model_path,
                model= model
            )
            return model_trainer_artifacts

        except Exception as e:
            lg.error(e)
            raise e