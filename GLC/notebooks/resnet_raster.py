import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np

class MultiModalTransform(object):
    def __init__(self):
        self.norm_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __call__(self, tensor):
        tensor = tensor.to(torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0) 
        
        tensor[tensor < -10000.0] = 0.0
        tensor[tensor > 10000.0] = 0.0
        
        if tensor.shape[0] >= 3:
            tensor[:3, :, :] = self.norm_rgb(tensor[:3, :, :])
        return tensor

transform_multimodal = transforms.Compose([MultiModalTransform()])

class GeoLifeMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(GeoLifeMultimodalModel, self).__init__()
        
        self.visual_branch = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.visual_branch.fc = nn.Identity()
        
        self.env_branch = nn.Sequential(
            nn.BatchNorm1d(30),
            nn.Linear(30, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048 + 256, num_classes)
        )

    def forward(self, x):
        img_rgb = x[:, :3, :, :]
        env_spatial = x[:, 3:, :, :]
        
        vis_features = self.visual_branch(img_rgb)
        
        env_vector = env_spatial.mean(dim=[2, 3])
        env_features = self.env_branch(env_vector)
        
        combined = torch.cat((vis_features, env_features), dim=1)
        out = self.classifier(combined)
        
        return out

def filter_out_of_bounds(dataset, extractor):
    keep_indices = []
    for i in tqdm(range(len(dataset)), desc=f"Nettoyage GPS {dataset.subset}"):
        try:
            lat, lon = dataset.coordinates[i]
            _ = extractor[(lat, lon)]
            keep_indices.append(i)
        except ValueError:
            pass
            
    dataset.observation_ids = dataset.observation_ids[keep_indices]
    dataset.coordinates = dataset.coordinates[keep_indices]
    if dataset.targets is not None:
        dataset.targets = dataset.targets[keep_indices]

def main():
    BASE_PATH = Path("..") 
    DATA_PATH = Path(r"C:\Users\Royann\Desktop\GeoLifeCLEF-GLC21\data")
    
    SCRATCH_PATH = Path(".")
    PATCHES_PATH = DATA_PATH / "patches"
    MODELS_PATH = BASE_PATH / "models"
    os.makedirs(MODELS_PATH, exist_ok=True)

    glc_path = str(SCRATCH_PATH)
    if glc_path not in sys.path:
        sys.path.append(glc_path)

    from GLC.data_loading.pytorch_dataset import GeoLifeCLEF2021Dataset
    from GLC.data_loading.environmental_raster import PatchExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nAppareil utilise : {device}")

    extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    extractor.add_all_rasters(nan=0.0001, out_of_bounds="ignore")

    train_dataset = GeoLifeCLEF2021Dataset(
        root=DATA_PATH, subset="train", use_rasters=True, 
        patch_extractor=extractor, transform=transform_multimodal
    )
    val_dataset = GeoLifeCLEF2021Dataset(
        root=DATA_PATH, subset="val", use_rasters=True, 
        patch_extractor=extractor, transform=transform_multimodal
    )

    valid_files = list(PATCHES_PATH.rglob("*_rgb.jpg"))
    valid_ids = set([int(f.stem.split('_')[0]) for f in valid_files])

    train_indices = [i for i, obs_id in enumerate(train_dataset.observation_ids) if obs_id in valid_ids]
    train_dataset.observation_ids = train_dataset.observation_ids[train_indices]
    train_dataset.coordinates = train_dataset.coordinates[train_indices]
    if train_dataset.targets is not None: 
        train_dataset.targets = train_dataset.targets[train_indices]

    val_indices = [i for i, obs_id in enumerate(val_dataset.observation_ids) if obs_id in valid_ids]
    val_dataset.observation_ids = val_dataset.observation_ids[val_indices]
    val_dataset.coordinates = val_dataset.coordinates[val_indices]
    if val_dataset.targets is not None: 
        val_dataset.targets = val_dataset.targets[val_indices]

    filter_out_of_bounds(train_dataset, extractor)
    filter_out_of_bounds(val_dataset, extractor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    vrai_num_classes = int(train_dataset.targets.max()) + 1

    model = GeoLifeMultimodalModel(num_classes=vrai_num_classes).to(device)

    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            print(f"\nEpoque {epoch+1}/{num_epochs}")
            
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc="Entrainement", leave=False)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)
            
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            
            val_pbar = tqdm(val_loader, desc="Validation  ", leave=False)
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * correct / total
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
        return history

    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    start_total_time = time.time()
    
    history = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    torch.save(model.state_dict(), MODELS_PATH / "resnet50_geolife_multimodal.pth")

    epochs_range = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title("Perte")
    ax1.legend()
    ax1.grid(True, linestyle='--')

    ax2.plot(epochs_range, history['val_accuracy'], label='Val Accuracy', marker='s', color='green')
    ax2.set_title("Precision")
    ax2.legend()
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig("courbes_resnet50_multimodal.png")

if __name__ == "__main__":
    main()