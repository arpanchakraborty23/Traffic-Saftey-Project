ARTIFACTS_FOLDER:str= 'Artifacts'
DATASET_URL:str ='https://universe.roboflow.com/ds/zEPfaXDzeb?key=LDZMbt7Cmb'

"Define Data Ingation constants"
DATA_INGESTION_DIR:str = 'data_ingestion'
RAW_DATA_DIR:str = 'DATASET.zip'
FEATURE_STORE_Dir:str = 'FEATURE_STORE'

"Define Data Validation"
DATA_VALIDATION_DIR_NAME:str="data_validation"

DATA_VALIDATION_STATUS_FILE:str="status.txt"
DATA_VALIDATION_ALL_REQUIRED_FILES:list[str]=["train","test","valid"]   

"Define Model Train constants"
MODEL_TRAIN_DIR:str = 'model_trainer'
PRE_TRAIN_MODEL_FILE:str = 'yolo12n.pt'
TRAINED_MODEL_OUTPUT_PATH:str ='results'
TRAINED_MODEL_PATH:str = 'Trained_model.pt'
TRAIN_EPOCHS:int = 10
BATCH_SIZE :int = 16
