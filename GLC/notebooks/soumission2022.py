import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm


class Transform:
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, patches):
        rgb = patches[0]
        nir = patches[1]
        env = patches[-1] 
        
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb.copy()).float() / 255.0
            if rgb.shape[-1] == 3: rgb = rgb.permute(2, 0, 1)
                
        if isinstance(nir, np.ndarray):
            nir = torch.from_numpy(nir.copy()).float() / 255.0
            if nir.ndim == 2: nir = nir.unsqueeze(0)
            elif nir.shape[-1] == 1: nir = nir.permute(2, 0, 1)
            
        # Création de l'image NIR + G + B
        img_bimodal = torch.cat([nir, rgb[1:3, :, :]], dim=0)
        img_bimodal = self.normalize(img_bimodal)

        # Climat
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env.copy()).float()
            
        env = torch.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        env[(env < -10000.0) | (env > 10000.0)] = 0.0
        env_vector = env.mean(dim=[1, 2]) 
        
        return img_bimodal, env_vector


class SafeDataset(torch.utils.data.Dataset):
    """Protège contre les images manquantes et les rasters hors bordure"""
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except (FileNotFoundError, ValueError): 
            new_index = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(new_index)


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.visual_backbone = models.resnet34(weights=None)
        self.visual_backbone.fc = nn.Identity()
        
        self.env_backbone = nn.Sequential(
            nn.BatchNorm1d(27),
            nn.Linear(27, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 256, num_classes)
        )

    def forward(self, img, env):
        vis_feat = self.visual_backbone(img)
        env_feat = self.env_backbone(env)
        combined = torch.cat((vis_feat, env_feat), dim=1)
        return self.classifier(combined)


def main():
    BASE_PATH = Path("..") 
    DATA_PATH = Path(r"C:\Users\Royann\Desktop\GeoLifeCLEF-GLC21\data22")
    MODELS_PATH = BASE_PATH / "models"
    SCRATCH_PATH = Path(".")

    glc_path = str(SCRATCH_PATH)
    if glc_path not in sys.path:
        sys.path.append(glc_path)

    from GLC.data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset    #utiliser le dataset de 2022
    from GLC.data_loading.environmental_raster import PatchExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Appareil utilisé : {device}")

    print(" Chargement de l'extracteur de Rasters...")
    extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    extractor.add_all_rasters(nan=0.0001, out_of_bounds="ignore")

    print(" Chargement des données de validation 2022 (France uniquement)...")
    raw_val_dataset = GeoLifeCLEF2022Dataset(
        root=DATA_PATH, subset="val", region="fr", patch_data="all", use_rasters=True, 
        patch_extractor=extractor, transform=Transform(is_train=False)
    )

    val_dataset = SafeDataset(raw_val_dataset)
    print(f" Total d'images à évaluer : {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    print(" Calcul automatique du nombre de classes pour la France...")
    temp_train_dataset = GeoLifeCLEF2022Dataset(root=DATA_PATH, subset="train", region="fr", use_rasters=False)
    num_classes_fr = int(temp_train_dataset.targets.max()) + 1
    print(f" Nombre de classes détectées : {num_classes_fr}")

    print(" Initialisation et chargement du Modèle...")
    model = Model(num_classes=num_classes_fr).to(device)
    
    fichier_modele = MODELS_PATH / "resnet34_fr_final.pth" 
    
    if not fichier_modele.exists():
        print(f" ERREUR : Le modèle {fichier_modele.name} n'a pas été trouvé.")
        return
        
    model.load_state_dict(torch.load(fichier_modele, map_location=device))
    print(f" Modèle chargé avec succès !")

    print("\n Début du calcul du score Top-30...")
    model.eval() 
    
    correct_top1 = 0
    correct_top30 = 0
    total = 0
    
    val_pbar = tqdm(val_loader, desc="Évaluation")
    
    with torch.no_grad():
        for inputs, labels in val_pbar:
            img_bimodal, env_vec = inputs[0].to(device), inputs[1].to(device)
            labels = labels.to(device)
            
            outputs = model(img_bimodal, env_vec)
            
            # CALCUL DU TOP-30
            _, pred_top30 = outputs.topk(30, dim=1, largest=True, sorted=True)
            correct_top30 += pred_top30.eq(labels.view(-1, 1).expand_as(pred_top30)).sum().item()
            
            # CALCUL DU TOP-1
            _, predicted = torch.max(outputs.data, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            total += labels.size(0)
            
            # Affichage en direct sur la barre de progression
            acc_1 = 100 * correct_top1 / total
            acc_30 = 100 * correct_top30 / total
            val_pbar.set_postfix({'Top-1': f"{acc_1:.1f}%", 'Top-30': f"{acc_30:.1f}%"})

    final_top1 = 100 * correct_top1 / total
    final_top30 = 100 * correct_top30 / total
    
    print("\n=======================================================")
    print(f" RÉSULTATS FINAUX SUR LE JEU 2022 (Validation France) :")
    print(f"   Précision    : {final_top1:.2f}%")
    print(f"   Top-30       : {final_top30:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    main()
