from src.configuration.traning_config import DataConfiguration,DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.logger import logging as lg

class TraningPipline:
    def __init__(self):
        self.traning_config = DataConfiguration()
    
    def start_data_ingestion(self):
        lg.info('******************************** Data Ingestion ***********************************')
        data_ingestion_config = DataIngestionConfig(traning_config=self.traning_config)
        data_ingestion= DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
        print('data ingestion completed')
        lg.info('******************************** Data Ingestion Completed *******************************')

    def run_pipline(self):
        self.start_data_ingestion()

if __name__=="__main__":
    obj=TraningPipline()
    obj.run_pipline()