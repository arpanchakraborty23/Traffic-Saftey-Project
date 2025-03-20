from src.configuration.traning_config import DataConfiguration,DataIngestionConfig,DataValidationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
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

    def run_pipline(self):
        self.start_data_ingestion()
        self.start_data_validation()

if __name__=="__main__":
    obj=TraningPipline()
    obj.run_pipline()