from src.logger import logging as lg
from src.configuration.traning_config import ModelTrainerConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts,DataValidationArtifact,DataIngestionArtifacts
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml
import os

class ModelTrainer:
    def __init__(self,
                model_config:ModelTrainerConfig,
                ingestion_artifact:DataIngestionArtifacts,
                validation_artifact:DataValidationArtifact):
        self.model_train_config = model_config
        self.ingestion_artifacts = ingestion_artifact
        self.validation_artifact = validation_artifact

        
    def fine_tuning(self,model,data_file_path,batch_size,epochs):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model =model.to(device)

            # Define train parameters
            lr = 0.001
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()

            # apply Quantization on pretrain model
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model= model,
                qconfig_spec={torch.nn.Linear},
                dtype= torch.qint8
            ).to(device)
            
            # Data normalization config
            transform = transforms.Compose([
            transforms.Resize((128, 128)),    # Resize to 128x128
            transforms.RandomHorizontalFlip(),  # Randomly flip image
            transforms.ToTensor(),               # Convert to tensor (HWC → CHW, 0-255 → 0-1)
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
            ])

            # Load custom dataset (train/val)
            train_data_loader = DataLoader(data_file_path['train'],batch_size=16, shuffle=True)
            val_data_loader = DataLoader(data_file_path['val'],batch_size=16, shuffle=False)

            # start fine tuning model
            for epoch in range(epochs):
                quantized_model.train()
                train_loss= 0

                for images, target in train_data_loader:
                    images, target = images.to(device),target.to(device)
                    optimizer.zero_grad()
                    outputs = quantized_model(images)
                    loss = criterion(outputs,target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss/ len(train_data_loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                lg.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            # Validation loop
            quantized_model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, targets in val_data_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs = quantized_model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == targets).sum().item()

            val_loss /= len(val_data_loader)
            val_accuracy = correct / len(data_file_path['val'])
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
            lg.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")
                        
        except Exception as e:
            lg.error(e)
            raise e


    def initate_model_trainer(self):
        try:
            # get dataset path
            dir_path = self.ingestion_artifacts.unzip_data_path
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f'{dir_path} not found')
            else:
                print(f'Dataset path founded at {dir_path}')
                lg.info(f'Dataset path founded at {dir_path}')

            # get  data.yaml
            for file in os.listdir(dir_path):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    data_file_path = os.path.join(dir_path,file)

            # loading Yolov12N 2.6 Milon params
            model = YOLO(self.model_train_config.pre_trained_model_path)
            lg.info("Model loaded successfully.")

            # Check if training directory exists and create it if not
            training_dir = data_file_path
            if not os.path.exists(training_dir):
                raise FileNotFoundError(f"Training directory {training_dir} not found.")
            lg.info(f"Training data directory found at {training_dir}")

            # Start model traning
            lg.info('Start model Traning ...')
            model.train(
                data= data_file_path,
                epochs = self.model_train_config.num_epochs,
                batch = self.model_train_config.batch_size,
                project = self.model_train_config.outputs_path
            )
            # save model
            model.save(self.model_train_config.model_path)
            lg.info(f"Trained model saved at: {self.model_train_config.model_path}")

            # Create and return ModelTrainingArtifact
            model_training_artifact = ModelTrainerArtifacts(
                model_path= self.model_train_config.model_path
            )
            lg.info("ModelTrainingArtifact created successfully.")
            return model_training_artifact
        except Exception as e :
            lg.error(e)
            raise e