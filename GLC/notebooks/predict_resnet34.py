import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
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
            
        img_bimodal = torch.cat([nir, rgb[1:3, :, :]], dim=0)
        img_bimodal = self.normalize(img_bimodal)

        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env.copy()).float()
            
        env = torch.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        env[(env < -10000.0) | (env > 10000.0)] = 0.0
        env_vector = env.mean(dim=[1, 2]) 
        
        return img_bimodal, env_vector

class SafeValDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        except (FileNotFoundError, ValueError): 
            img_bimodal = torch.zeros((3, 256, 256), dtype=torch.float32)
            env_dummy = torch.zeros(27, dtype=torch.float32)
            # Le jeu de validation attend toujours un label à la fin, on renvoie -1
            return (img_bimodal, env_dummy), -1

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

    from data_loading22.pytorch_dataset import GeoLifeCLEF2022Dataset    #Utiliser le dataset 2022
    from data_loading22.environmental_raster import PatchExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Appareil utilisé : {device}")

    print(" Chargement de l'extracteur de Rasters...")
    extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    extractor.add_all_rasters(nan=0.0001, out_of_bounds="ignore")

    print(" Chargement des données de VALIDATION (France)...")
    raw_val_dataset = GeoLifeCLEF2022Dataset(
        root=DATA_PATH, subset="val", region="fr", patch_data="all", use_rasters=True, 
        patch_extractor=extractor, transform=Transform(is_train=False)
    )

    val_dataset = SafeValDataset(raw_val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    temp_train = GeoLifeCLEF2022Dataset(root=DATA_PATH, subset="train", region="fr", use_rasters=False)
    num_classes_fr = int(temp_train.targets.max()) + 1

    print(" Chargement du Modèle...")
    model = Model(num_classes=num_classes_fr).to(device)
    model.load_state_dict(torch.load(MODELS_PATH / "resnet34_fr_final.pth", map_location=device))
    model.eval() 

    kaggle_submission = []
    observation_ids = raw_val_dataset.observation_ids
    current_idx = 0

    print("\n Génération des prédictions ...")
    with torch.no_grad():
        for inputs, _ in tqdm(val_loader, desc="Inférence Soumission"):
            img, env = inputs[0].to(device), inputs[1].to(device)
            outputs = model(img, env)
            
            _, top30 = outputs.topk(30, dim=1, largest=True, sorted=True)
            top30_numpy = top30.cpu().numpy()
            
            for i in range(len(top30_numpy)):
                obs_id = observation_ids[current_idx]
                pred_string = " ".join(map(str, top30_numpy[i]))
                kaggle_submission.append({
                    "ObservationId": obs_id,
                    "Predicted_class": pred_string
                })
                current_idx += 1

    print("\n Sauvegarde des résultats...")
    df_submission = pd.DataFrame(kaggle_submission)
    df_submission.to_csv("submission_fr.csv", index=False)
    print(f" Fichier de prédictions généré avec succès : submission_fr.csv ({len(df_submission)} lignes)")

if __name__ == "__main__":
    main()
