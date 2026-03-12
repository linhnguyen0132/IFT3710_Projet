import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
import pandas as pd # NOUVEAU : Indispensable pour la sauvegarde du CSV

# --- 1. TRANSFORMATION MULTIMODALE (33 CANAUX) ---
class MultiModalTransform(object):
    def __init__(self):
        self.norm_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __call__(self, tensor):
        tensor = tensor.to(torch.float32)
        # Bouclier contre les valeurs vides ou infinies
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0) 
        tensor[tensor < -10000.0] = 0.0
        tensor[tensor > 10000.0] = 0.0
        
        # Normalisation sur le RGB uniquement
        if tensor.shape[0] >= 3:
            tensor[:3, :, :] = self.norm_rgb(tensor[:3, :, :])
        return tensor

transform_multimodal = transforms.Compose([MultiModalTransform()])

# --- 2. ARCHITECTURE DU MODÈLE MULTIMODAL ---
class GeoLifeMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(GeoLifeMultimodalModel, self).__init__()
        
        self.visual_branch = models.resnet50(weights=None) # weights=None car on va charger vos poids !
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

# --- 3. FONCTION DE NETTOYAGE GPS ---
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
    return keep_indices

# --- SCRIPT PRINCIPAL ---
def main():
    BASE_PATH = Path("..") 
    DATA_PATH = Path(r"C:\Users\Royann\Desktop\GeoLifeCLEF-GLC21\data")
    SCRATCH_PATH = Path(".")
    
    glc_path = str(SCRATCH_PATH)
    if glc_path not in sys.path:
        sys.path.append(glc_path)

    from GLC.data_loading.pytorch_dataset import GeoLifeCLEF2021Dataset
    from GLC.data_loading.environmental_raster import PatchExtractor
    from GLC.metrics import top_30_error_rate, predict_top_30_set, top_k_error_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    print("🌍 Chargement de l'extracteur de Rasters...")
    extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    extractor.add_all_rasters(nan=0.0001, out_of_bounds="ignore")

    print("📦 Chargement des données (Rasters ACTIVÉS)...")
    # NOUVEAU : use_rasters=True pour tous les datasets
    train_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="train", use_rasters=True, transform=transform_multimodal, patch_extractor=extractor)
    val_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="val", use_rasters=True, transform=transform_multimodal, patch_extractor=extractor)
    test_dataset = GeoLifeCLEF2021Dataset(root=DATA_PATH, subset="test", use_rasters=True, transform=transform_multimodal, patch_extractor=extractor)

    print("Recherche des images disponibles sur le disque local...")
    PATCHES_PATH = DATA_PATH / "patches"
    valid_files = list(PATCHES_PATH.rglob("*_rgb.jpg"))
    valid_ids = set([int(f.stem.split('_')[0]) for f in valid_files])

    # Filtrer le jeu d'Entraînement
    train_indices = [i for i, obs_id in enumerate(train_dataset.observation_ids) if obs_id in valid_ids]
    train_dataset.observation_ids = train_dataset.observation_ids[train_indices]
    train_dataset.coordinates = train_dataset.coordinates[train_indices]
    if train_dataset.targets is not None: 
        train_dataset.targets = train_dataset.targets[train_indices]

    # Filtrer le jeu de Validation
    val_indices = [i for i, obs_id in enumerate(val_dataset.observation_ids) if obs_id in valid_ids]
    val_dataset.observation_ids = val_dataset.observation_ids[val_indices]
    val_dataset.coordinates = val_dataset.coordinates[val_indices]
    if val_dataset.targets is not None: 
        val_dataset.targets = val_dataset.targets[val_indices]

    # Filtrer le jeu de Test
    test_indices = [i for i, obs_id in enumerate(test_dataset.observation_ids) if obs_id in valid_ids]
    test_dataset.observation_ids = test_dataset.observation_ids[test_indices]
    test_dataset.coordinates = test_dataset.coordinates[test_indices]
    
    # NOUVEAU : Nettoyage GPS des hors-limites
    filter_out_of_bounds(train_dataset, extractor)
    filter_out_of_bounds(val_dataset, extractor)
    filter_out_of_bounds(test_dataset, extractor)
    
    print(f"Images d'entraînement conservées : {len(train_dataset)}")
    print(f"Images de validation conservées : {len(val_dataset)}")
    print(f"Images de test conservées : {len(test_dataset)}")

    # SÉCURITÉ WINDOWS : num_workers=0 pour éviter les crashs silencieux
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    vrai_num_classes = 31187

    print("\n🧠 Chargement du Modèle Multimodal...")
    # 1. On charge l'architecture personnalisée
    model = GeoLifeMultimodalModel(num_classes=vrai_num_classes)
    
    # 2. On charge vos poids fraîchement entraînés
    model_path = Path(r"C:\Users\Royann\Desktop\GeoLifeCLEF-GLC21\models\resnet50_geolife_multimodal.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    model.eval() # On passe en mode évaluation (désactive le dropout, fige la batchnorm)

    # --- 1. EVALUATION SUR LE JEU D'ENTRAÎNEMENT ---
    print("\nÉvaluation sur le jeu d'entraînement...")
    all_train_preds = []
    all_train_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(train_loader, desc="Train Eval"):
            outputs = model(inputs.to(device))
            all_train_preds.append(outputs.cpu().numpy())
            all_train_labels.append(labels.numpy())

    if len(all_train_preds) > 0:
        y_score_train = np.concatenate(all_train_preds)
        y_true_train = np.concatenate(all_train_labels)

        # Calcul Top-30
        error_rate_30_train = top_30_error_rate(y_true_train, y_score_train)
        top30_acc_train = (1 - error_rate_30_train) * 100
        
        # Calcul Top-1 (Accuracy basique)
        error_rate_1_train = top_k_error_rate(y_true_train, y_score_train, k=1)
        top1_acc_train = (1 - error_rate_1_train) * 100

        print(f"📊 Train - Accuracy Basique (Top-1) : {top1_acc_train:.2f}%")
        print(f"📊 Train - Top-30 Accuracy         : {top30_acc_train:.2f}%")

    # --- 2. EVALUATION SUR LE JEU DE VALIDATION ---
    print("\nÉvaluation sur le jeu de validation...")
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation Eval"):
            outputs = model(inputs.to(device))
            all_val_preds.append(outputs.cpu().numpy())
            all_val_labels.append(labels.numpy())

    if len(all_val_preds) > 0:
        y_score_val = np.concatenate(all_val_preds)
        y_true_val = np.concatenate(all_val_labels)

        # Calcul Top-30
        error_rate_30_val = top_30_error_rate(y_true_val, y_score_val)
        top30_acc_val = (1 - error_rate_30_val) * 100
        
        # Calcul Top-1 (Accuracy basique)
        error_rate_1_val = top_k_error_rate(y_true_val, y_score_val, k=1)
        top1_acc_val = (1 - error_rate_1_val) * 100

        print(f"✅ Val - Accuracy Basique (Top-1) : {top1_acc_val:.2f}%")
        print(f"✅ Val - Top-30 Accuracy          : {top30_acc_val:.2f}%\n")

    # --- 3. SOUMISSION ---
    print("Génération des prédictions pour la soumission...")
    all_test_preds = []
    
    with torch.no_grad():
        # Correction sécurité pour le jeu de test (gère tuple ou tenseur direct)
        for batch in tqdm(test_loader, desc="Test Inférence"): 
            inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
            outputs = model(inputs.to(device))
            all_test_preds.append(outputs.cpu().numpy())

    if len(all_test_preds) > 0:
        y_score_test = np.concatenate(all_test_preds)
        
        top30_sets = predict_top_30_set(y_score_test)
        all_preds_str = [" ".join(map(str, row)) for row in top30_sets]

        submission_df = pd.DataFrame({
            'ObservationId': test_dataset.observation_ids,
            'Predicted_SpeciesId': all_preds_str
        })

        submission_df.to_csv("submission.csv", index=False, sep=',')
        print("🎉 Fichier 'submission.csv' généré avec succès !")
    else:
        print("⚠️ Aucune donnée de test n'a pu être évaluée.")

if __name__ == "__main__":
    main()