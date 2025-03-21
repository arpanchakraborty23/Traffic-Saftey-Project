from src.configuration.traning_config import DataConfiguration,DataIngestionConfig,DataValidationConfig,ModelTrainerConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.logger import logging as lg

class TraningPipline:
    def __init__(self):
        self.traning_config = DataConfiguration()
    
    def start_data_ingestion(self):
        lg.info('******************************** Data Ingestion ***********************************')
        data_ingestion_config = DataIngestionConfig(traning_config=self.traning_config)
        data_ingestion= DataIngestion(config=data_ingestion_config)
        self.data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
        print('data ingestion completed')
        lg.info('******************************** Data Ingestion Completed *******************************')
       
    
    def start_data_validation(self):
        lg.info('******************************** Data Validation ***********************************')
        data_validation_config=DataValidationConfig(traning_config=self.traning_config)
        data_validation=DataValidation(
            validation_config=data_validation_config,
            ingestion_artifacts=self.data_ingestion_artifacts
        )
        self.data_validation_artifacts=data_validation.initiate_data_val()
        print('Data validation completed')
        lg.info('******************************** Data Validation Completed *******************************')
    
    def start_model_training(self):
            lg.info('******************************** Model Training ***********************************')
            model_training_config=ModelTrainerConfig(traning_config=self.traning_config)
            model_training= ModelTrainer(
                ingestion_artifact= self.data_ingestion_artifacts,
                validation_artifact=self.data_validation_artifacts,
                model_config=model_training_config
            )
            model_training.initate_model_trainer()
            lg.info('******************************** Model Training Completed *******************************')
       

    def run_pipline(self):
        self.start_data_ingestion()
        self.start_data_validation()
        self.start_model_training()

if __name__=="__main__":
    obj=TraningPipline()
    obj.run_pipline()