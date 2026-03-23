import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 

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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Appareil utilisé : {device}")


    print(" Chargement des données...")
    train_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="train", use_rasters=False)
    val_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="val", use_rasters=False)

    valid_files = list(PATCHES_PATH.rglob("*_rgb.jpg"))
    valid_ids = set([int(f.stem.split('_')[0]) for f in valid_files])

    # Filtre Train
    train_indices = [i for i, obs_id in enumerate(train_dataset.observation_ids) if obs_id in valid_ids]
    train_dataset.observation_ids = train_dataset.observation_ids[train_indices]
    train_dataset.coordinates = train_dataset.coordinates[train_indices]
    if train_dataset.targets is not None: 
        train_dataset.targets = train_dataset.targets[train_indices]

    # Filtre Val
    val_indices = [i for i, obs_id in enumerate(val_dataset.observation_ids) if obs_id in valid_ids]
    val_dataset.observation_ids = val_dataset.observation_ids[val_indices]
    val_dataset.coordinates = val_dataset.coordinates[val_indices]
    if val_dataset.targets is not None: 
        val_dataset.targets = val_dataset.targets[val_indices]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    vrai_num_classes = int(train_dataset.targets.max()) + 1

    print(f" Données prêtes ! Nombre de classes : {vrai_num_classes}")


    print(" Initialisation de ResNet-50...")

    model = models.resnet50(weights=None)
    
    #Modifier la première couche pour accepter 6 canaux au lieu de 3
    #Les autres paramètres (kernel_size=7, stride=2, padding=3) restent standards pour ResNet
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    #Modifier la dernière couche (fully connected) pour le nombre de classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, vrai_num_classes)
    
    model = model.to(device)


    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(num_epochs):
            print(f"\nÉpoque {epoch+1}/{num_epochs}")
            
            # Phase d'entraînement
            model.train()
            train_loss = 0.0
            
            # Barre de progression pour l'entraînement
            train_pbar = tqdm(train_loader, desc="Entraînement", leave=False)
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                # Met à jour l'affichage de la barre avec la perte courante
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)
            
            # Phase d'évaluation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            
            # Barre de progression pour la validation
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

    print("\n Début de l'entraînement de ResNet-50...")
    start_total_time = time.time()
    history = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    print(f"\n Entraînement terminé en {(time.time() - start_total_time)/60:.1f} minutes.")


    torch.save(model.state_dict(), MODELS_PATH / "resnet50_geolife.pth")
    print(" Modèle sauvegardé avec succès")

    epochs_range = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs_range, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title("Perte")
    ax1.legend()
    ax1.grid(True, linestyle='--')

    ax2.plot(epochs_range, history['val_accuracy'], label='Val Accuracy', marker='s', color='green')
    ax2.set_title("Précision")
    ax2.legend()
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig("courbes_resnet50.png")
    print(" Graphique sauvegardé sous 'courbes_resnet50.png'")

if __name__ == "__main__":

    main()
