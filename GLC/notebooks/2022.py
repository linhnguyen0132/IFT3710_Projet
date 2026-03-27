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
import random

class SafeDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            return self.dataset[index]
        # On attrape la FileNotFoundError (image absente) ET la ValueError (raster hors bordure)
        except (FileNotFoundError, ValueError): 
            # On ignore l'erreur et on pioche une autre plante au hasard !
            new_index = random.randint(0, len(self.dataset) - 1)
            return self.__getitem__(new_index)


class Transform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        # Augmentations Luminosité, Contraste
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        # Normalisation ImageNet classique
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, patches):
        # Extraction (GeoLife2022 renvoie : [RGB, Near_IR, LandCover, Altitude, Environnement])
        rgb = patches[0]
        nir = patches[1]
        env = patches[-1]
        
        # Formatage RGB (devient 3, 256, 256)
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb.copy()).float() / 255.0
            if rgb.shape[-1] == 3: rgb = rgb.permute(2, 0, 1)
                
        # Formatage NIR (devient 1, 256, 256)
        if isinstance(nir, np.ndarray):
            nir = torch.from_numpy(nir.copy()).float() / 255.0
            if nir.ndim == 2: nir = nir.unsqueeze(0)
            elif nir.shape[-1] == 1: nir = nir.permute(2, 0, 1)

        # On crée une image à 3 canaux (Infrarouge, Vert, Bleu)
        # On remplace le canal Rouge (index 0 de rgb) par le canal Infrarouge (nir)
        img_bimodal = torch.cat([nir, rgb[1:3, :, :]], dim=0)

        # Application des augmentations d'images
        if self.is_train:
            if torch.rand(1).item() > 0.5: img_bimodal = transforms.functional.hflip(img_bimodal)
            if torch.rand(1).item() > 0.5: img_bimodal = transforms.functional.vflip(img_bimodal)
            # Rotation aléatoire entre -15° et 15°
            angle = torch.empty(1).uniform_(-15, 15).item()
            img_bimodal = transforms.functional.rotate(img_bimodal, angle)
            img_bimodal = self.color_jitter(img_bimodal)
            
        img_bimodal = self.normalize(img_bimodal)

        # Formatage du climat (Sécurisé contre les valeurs infinies)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env.copy()).float()
            
        env = torch.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
        env[(env < -10000.0) | (env > 10000.0)] = 0.0
        
        # On moyenne la zone pour obtenir un simple vecteur de 30 valeurs environnementales
        env_vector = env.mean(dim=[1, 2]) 
        
        return img_bimodal, env_vector


# LE MODÈLE ResNet-34 + Réseau Environnemental

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        # Branche 1 : ResNet34
        self.visual_backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.visual_backbone.fc = nn.Identity() # Sort un vecteur de taille 512
        
        # Branche 2 : MLP Environnemental
        self.env_backbone = nn.Sequential(
            nn.BatchNorm1d(27),
            nn.Linear(27, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Fusion Finale
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 256, num_classes) # 512 (Vue) + 256 (Climat)
        )

    def forward(self, img, env):
        vis_feat = self.visual_backbone(img)
        env_feat = self.env_backbone(env)
        
        combined = torch.cat((vis_feat, env_feat), dim=1)
        return self.classifier(combined)



def main():
    BASE_PATH = Path("..") 
    DATA_PATH = Path(r"C:\Users\Royann\Desktop\GeoLifeCLEF-GLC21\data22")
    
    SCRATCH_PATH = Path(".")
    MODELS_PATH = BASE_PATH / "models"
    os.makedirs(MODELS_PATH, exist_ok=True)

    glc_path = str(SCRATCH_PATH)
    if glc_path not in sys.path:
        sys.path.append(glc_path)

    from GLC.data_loading.pytorch_dataset import GeoLifeCLEF2022Dataset   #Utiliser le dataset de 2022
    from GLC.data_loading.environmental_raster import PatchExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Appareil utilisé : {device}")

    print(" Chargement de l'extracteur de Rasters...")
    extractor = PatchExtractor(DATA_PATH / "rasters", size=256)
    extractor.add_all_rasters(nan=0.0001, out_of_bounds="ignore")

    print(" Chargement des données (France uniquement)...")
    
    raw_train_dataset = GeoLifeCLEF2022Dataset(
        root=DATA_PATH, subset="train", region="fr", patch_data="all", use_rasters=True, 
        patch_extractor=extractor, transform=Transform(is_train=True)
    )
    raw_val_dataset = GeoLifeCLEF2022Dataset(
        root=DATA_PATH, subset="val", region="fr", patch_data="all", use_rasters=True, 
        patch_extractor=extractor, transform=Transform(is_train=False)
    )

    train_dataset = SafeDataset(raw_train_dataset)
    val_dataset = SafeDataset(raw_val_dataset)

    # On calcule le nombre de classes 
    vrai_num_classes = int(raw_train_dataset.targets.max()) + 1
    print(f" Dataset filtré sur la France : {len(train_dataset)} images, {vrai_num_classes} classes.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    print(" Initialisation du Modèle (ResNet34 + Environnement)...")
    model = Model(num_classes=vrai_num_classes).to(device)

    # =================================================================
    # OPTIONNEL : SYSTÈME DE REPRISE DE L'ENTRAÎNEMENT
    # Si le fichier existe, le modèle reprend là où il s'est arrêté 
    # =================================================================
    fichier_reprise = MODELS_PATH / "resnet34_fr_epoch_1.pth"
    start_epoch = 0
    
    if fichier_reprise.exists():
        print(f" Checkpoint trouvé ! Chargement depuis {fichier_reprise.name}...")
        model.load_state_dict(torch.load(fichier_reprise, map_location=device))
        print(" Modèle rechargé. L'entraînement va continuer !")
        start_epoch = 1
    else:
        print("Aucun checkpoint trouvé. Le modèle commence de zéro.")

    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, start_epoch=0):
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n🔄 Époque {epoch+1}/{num_epochs}")
            
            # Phase d'entraînement
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc="Entraînement", leave=False)
            for inputs, labels in train_pbar:
                img_bimodal, env_vec = inputs[0].to(device), inputs[1].to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(img_bimodal, env_vec)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Le limiteur de vitesse pour éviter les crashs (NaN)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()
                
                train_loss += loss.item() * labels.size(0)
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)
            
            # Phase d'évaluation
            model.eval()
            val_loss, correct, total = 0.0, 0, 0
            
            val_pbar = tqdm(val_loader, desc="Validation  ", leave=False)
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    img_bimodal, env_vec = inputs[0].to(device), inputs[1].to(device)
                    labels = labels.to(device)
                    
                    outputs = model(img_bimodal, env_vec)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100 * correct / total
            
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            print(f" Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
            # SAUVEGARDE À CHAQUE ÉPOQUE
            checkpoint_path = MODELS_PATH / f"resnet34_fr_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Checkpoint sauvegardé : {checkpoint_path.name}")
            
        return history

    num_epochs = 10 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("\n Début de l'entraînement...")
    start_total_time = time.time()
    history = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch)
    print(f"\n Entraînement terminé en {(time.time() - start_total_time)/60:.1f} minutes.")

    # Sauvegarde finale
    torch.save(model.state_dict(), MODELS_PATH / "resnet34_fr_final.pth")
    print(" Modèle final sauvegardé avec succès")

    epochs_range = range(start_epoch + 1, num_epochs + 1)
    if len(epochs_range) > 0:
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
        plt.savefig("courbes_sensio_fr.png")
        print(" Graphique sauvegardé sous 'courbes_sensio_fr.png'")

if __name__ == "__main__":
    main()
