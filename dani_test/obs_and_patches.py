import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

class GeoLifeDataset(Dataset):

    def __init__(self, root, subset="train", transform=None):

        self.root = Path(root)
        self.transform = transform

        print("Loading observations...")

        # Charger les données de la France
        df_fr = pd.read_csv(self.root / "observations" / "observations_fr_train.csv", sep=";")

        # Charger les données US
        df_us = pd.read_csv(self.root / "observations" / "observations_us_train.csv", sep=";")

        # Mettre US et France en un seul dataframe
        df = pd.concat([df_fr, df_us])

        # garder seulement les espèces avec >= 5 images
        valid_species = species_counts[species_counts >= 20].index

        # filtrer le dataframe
        df = df[df["species_id"].isin(valid_species)].reset_index(drop=True)

        # Créer un mapping des species label
        species_ids = sorted(df["species_id"].unique())
        self.label_map = {species_id: i for i, species_id in enumerate(species_ids)}

        # Split en train et validation
        df = df[df["subset"] == subset]

        patches_dir = self.root / "patches_sample"

        self.observations = []

        print("Building dataset...")

        for row in df.itertuples():  

            obs_id = row.observation_id
            species = row.species_id

            region = "fr" if obs_id < 20000000 else "us"

            sub1 = str(obs_id)[-2:]
            sub2 = str(obs_id)[-4:-2]

            img_path = patches_dir / region / sub1 / sub2 / f"{obs_id}_rgb.jpg"

            if img_path.exists():
                self.observations.append((img_path, species))

        print("Images trouvées :", len(self.observations))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):

        img_path, species = self.observations[idx]

        # récupérer l'id de l'observation
        obs_id = img_path.stem.replace("_rgb", "")

        # chemin vers l'image NIR
        nir_path = img_path.with_name(f"{obs_id}_near_ir.jpg")

        # charger RGB
        rgb = Image.open(img_path).convert("RGB")

        # charger NIR (grayscale)
        if nir_path.exists():
            nir = Image.open(nir_path).convert("L")
        else:
            print("NIR not found!")
        
        # convertir en numpy
        rgb = np.array(rgb)
        nir = np.array(nir)

        # ajouter une dimension au NIR
        nir = nir[:, :, None]

        # concaténer RGB + NIR → 4 canaux
        image = np.concatenate([rgb, nir], axis=2)

        # convertir en tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255

        if self.transform:
            image = self.transform(image)

        label = self.label_map[species]

        return image, label