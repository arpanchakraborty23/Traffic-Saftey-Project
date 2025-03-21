from src.logger import logging as lg
from src.configuration.traning_config import ModelTrainerConfig
from src.entity.artifacts_entity import ModelTrainerArtifacts, DataValidationArtifact, DataIngestionArtifacts
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import yaml
import os
import traceback
from PIL import Image
import mlflow
import mlflow.pytorch

class YoloDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.label_paths = []
        self.transform = transform

        # Traverse all subfolders to collect images and corresponding labels
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    label_path = os.path.join(root, file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
                    if os.path.exists(label_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Read label file
        with open(self.label_paths[idx], 'r') as f:
            labels = f.readlines()
        labels = [list(map(float, line.strip().split())) for line in labels]

        return image, torch.tensor(labels)

class ModelTrainer:
    def __init__(self, model_config: ModelTrainerConfig, ingestion_artifact: DataIngestionArtifacts, validation_artifact: DataValidationArtifact):
        self.model_train_config = model_config
        self.ingestion_artifacts = ingestion_artifact
        self.validation_artifact = validation_artifact

    def read_yaml(self, path):
        with open(path, 'rb') as y:
            content = yaml.safe_load(y)
        return content

    def fine_tuning(self, model, data_file_path, batch_size, epochs):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # Define optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            # Data normalization & loading datasets
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            # Load custom YOLO dataset with support for nested directories
            train_dataset = YoloDataset(data_file_path['train'], transform=transform)
            val_dataset = YoloDataset(data_file_path['val'], transform=transform)

            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            with mlflow.start_run():
                mlflow.log_params({"batch_size": batch_size, "epochs": epochs})

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0

                    for images, targets in train_data_loader:
                        images = images.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()

                    avg_loss = train_loss / len(train_data_loader)
                    lg.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)

                # Post-training quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model=model,
                    qconfig_spec={torch.nn.Linear},
                    dtype=torch.qint8
                ).to(device)
                lg.info("Quantized model created successfully.")

                # Validation loop with mAP calculation
                model.eval()
                val_loss = 0
                correct = 0
                total_predictions = 0
                true_positives = 0
                false_positives = 0
                false_negatives = 0

                with torch.no_grad():
                    for images, targets in val_data_loader:
                        images = images.to(device)
                        outputs = quantized_model(images)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                        predictions = torch.argmax(outputs, dim=1)
                        correct += (predictions == targets).sum().item()
                        total_predictions += targets.size(0)

                        true_positives += ((predictions == 1) & (targets == 1)).sum().item()
                        false_positives += ((predictions == 1) & (targets == 0)).sum().item()
                        false_negatives += ((predictions == 0) & (targets == 1)).sum().item()

                val_loss /= len(val_data_loader)
                accuracy = correct / total_predictions
                precision = true_positives / (true_positives + false_positives + 1e-8)
                recall = true_positives / (true_positives + false_negatives + 1e-8)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

                mlflow.log_metrics({"val_loss": val_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score})
                mlflow.pytorch.log_model(model, "model")

                lg.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}")

        except Exception as e:
            lg.error(f"Error occurred: {e}")
            raise e

    def initate_model_trainer(self):
        try:
            # Get dataset path
            dir_path = self.ingestion_artifacts.unzip_data_path
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f'{dir_path} not found')
            lg.info(f'Dataset path found at {dir_path}')

            # Loop through all YAML files in a directory
            directory = self.ingestion_artifacts.unzip_data_path
            data_file_path = {}

            # Check all files in directory
            for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    _,file_name = os.path.split(file)
                    print(f"Found  file: {file_path}")

                    # Keep track of the first valid YAML file
                    data_file_path[file] = file_path
                    print(f" {data_file_path[file_name][-1]} , file path:{file_path}" )

                    if not data_file_path:
                        raise FileNotFoundError("No data.yaml file found in the directory.")
                    
                    lg.info(f"Contents of {data_file_path[file]} , file path:{file_path}" )
                    print()
                    print(data_file_path)
                    print()

            # Load YOLOv12 model
            model = YOLO(self.model_train_config.pre_trained_model_path)
            lg.info("Model loaded successfully.")

            # Start model training
            lg.info('Starting model training...')
            
            # model.train(
            #     data=data_file_path['data.yaml'],
            #     epochs=self.model_train_config.num_epochs,
            #     batch=self.model_train_config.batch_size,
            #     project=self.model_train_config.outputs_path
            # )
            
            model_path = self.fine_tuning(
                model=model,
                data_file_path=data_file_path,
                batch_size= self.model_train_config.batch_size,
                epochs= self.model_train_config.num_epochs

            )

            # # Save model
            # model.save(self.model_train_config.model_path)
            # lg.info(f"Trained model saved at: {self.model_train_config.model_path}")

            # Create and return ModelTrainingArtifact
            model_training_artifact = ModelTrainerArtifacts(
                model_path=model_path
            )
            lg.info("ModelTrainingArtifact created successfully.")
            return model_training_artifact

        except Exception as e:
            lg.error(f"Error occurred: {e}")
            raise e
