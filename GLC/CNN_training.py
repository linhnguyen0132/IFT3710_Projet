import sys
import os
import time
from pathlib import Path
import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# --- CORRECTION ABSOLUE DES CHEMINS ---
# 1. On identifie le dossier actuel (GLC)
dossier_actuel = os.path.dirname(os.path.abspath(__file__))

# 2. On remonte d'un seul niveau pour atteindre la racine (GLC-2021)
racine_projet = os.path.dirname(dossier_actuel)

# 3. On ajoute la racine en priorité absolue pour que Python trouve le module 'GLC'
if racine_projet not in sys.path:
    sys.path.insert(0, racine_projet)

from GLC.data_loading.pytorch_dataset import GeoLifeCLEF2021Dataset

# --- CONFIGURATION DYNAMIQUE ---
# Détection automatique du dossier de données (Local vs Cluster)
chemin_donnees_local = Path(r"C:\Users\abdou\Downloads\geolifeclef-2021\data")

if chemin_donnees_local.exists():
    # 1. Si on est sur ton PC, on utilise ton dossier de téléchargement
    DATA_PATH = chemin_donnees_local
    print("💻 Mode local détecté (Windows) : Données chargées depuis le dossier de téléchargement.")
else:
    # 2. Si on est sur le cluster Linux, on cherche 'data' à la racine du projet
    DATA_PATH = Path(racine_projet) / "data"
    print("🚀 Mode cluster détecté (Linux) : Données chargées depuis le dossier du projet.")

PATCHES_PATH = DATA_PATH / "patches_sample"
MODELS_PATH = Path(racine_projet) / "models"  # Les modèles seront toujours sauvegardés avec le code

# Détection automatique du GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil : {DEVICE}")

# --- MODÈLES ---
class SimpleGeoLifeCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleGeoLifeCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class BimodalGeoLifeCNN(nn.Module):
    def __init__(self, num_classes):
        super(BimodalGeoLifeCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_vis = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2_vis = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_vis = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv1_topo = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2_topo = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_topo = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*32*32 + 32*32*32, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x_vis, x_topo = x[:, :4, :, :], x[:, 4:, :, :]
        out_vis = self.pool(F.relu(self.conv3_vis(self.pool(F.relu(self.conv2_vis(self.pool(F.relu(self.conv1_vis(x_vis)))))))))
        out_topo = self.pool(F.relu(self.conv3_topo(self.pool(F.relu(self.conv2_topo(self.pool(F.relu(self.conv1_topo(x_topo)))))))))
        out = torch.cat((out_vis.view(out_vis.size(0), -1), out_topo.view(out_topo.size(0), -1)), dim=1)
        return self.fc2(self.dropout(F.relu(self.fc1(out))))

# --- FONCTIONS UTILES ---
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=2):
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        history['train_loss'].append(train_loss / len(train_loader.dataset))
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                total += labels.size(0)
                
        history['val_loss'].append(val_loss / len(val_loader.dataset))
        history['val_accuracy'].append(100 * correct / total)
        print(f"Epoque {epoch+1}/{num_epochs} [{time.time() - start_time:.1f}s] - Val Loss: {history['val_loss'][-1]:.4f} - Acc: {history['val_accuracy'][-1]:.2f}%")
        
    return history

def plot_results(history_uni, history_bi, num_epochs):
    epochs_range = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs_range, history_uni['val_loss'], label='Unimodal Loss', marker='o')
    ax1.plot(epochs_range, history_bi['val_loss'], label='Bimodal Loss', marker='s')
    ax1.set_title('Perte de Validation')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history_uni['val_accuracy'], label='Unimodal Acc', marker='o')
    ax2.plot(epochs_range, history_bi['val_accuracy'], label='Bimodal Acc', marker='s')
    ax2.set_title('Précision de Validation')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(MODELS_PATH / 'training_results.png')
    print(f"Graphique sauvegardé dans {MODELS_PATH}")

# --- BLOC PRINCIPAL ---
if __name__ == "__main__":
    # 1. Préparation des données
    print("Chargement des données...")
    train_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="train", use_rasters=False)
    val_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="val", use_rasters=False)

    valid_files = list(PATCHES_PATH.rglob("*_rgb.jpg"))
    valid_ids = set([int(f.stem.split('_')[0]) for f in valid_files])

    # Filtrage (Simplifié)
    for ds in [train_dataset, val_dataset]:
        indices = [i for i, obs_id in enumerate(ds.observation_ids) if obs_id in valid_ids]
        ds.observation_ids = ds.observation_ids[indices]
        ds.coordinates = ds.coordinates[indices]
        if ds.targets is not None: 
            ds.targets = ds.targets[indices]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    num_classes = int(train_dataset.targets.max()) + 1

    # 2. Entraînement
    num_epochs = 3
    criterion = nn.CrossEntropyLoss()

    print("\n🥊 ROUND 1 : Unimodal")
    model_uni = SimpleGeoLifeCNN(num_classes).to(DEVICE)
    optimizer_uni = optim.Adam(model_uni.parameters(), lr=0.001)
    history_uni = train_and_evaluate(model_uni, train_loader, val_loader, criterion, optimizer_uni, num_epochs)

    print("\n🥊 ROUND 2 : Bimodal")
    model_bi = BimodalGeoLifeCNN(num_classes).to(DEVICE)
    optimizer_bi = optim.Adam(model_bi.parameters(), lr=0.001)
    history_bi = train_and_evaluate(model_bi, train_loader, val_loader, criterion, optimizer_bi, num_epochs)

    # 3. Sauvegarde
    os.makedirs(MODELS_PATH, exist_ok=True)
    torch.save(model_uni.state_dict(), MODELS_PATH / "cnn_unimodal.pth")
    torch.save(model_bi.state_dict(), MODELS_PATH / "cnn_bimodal.pth")
    print("\n💾 Modèles sauvegardés.")

    # 4. Visualisation
    plot_results(history_uni, history_bi, num_epochs)