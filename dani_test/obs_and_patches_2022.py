import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


class GeoLifeDataset2(Dataset):

    def __init__(self, root, subset="train", transform=None, label_map=None):

        self.root = Path(root)
        self.transform = transform

        print("Loading observations...")

        df_fr = pd.read_csv(self.root / "observations" / "observations_fr_train.csv", sep=";")

        df = pd.concat([df_fr])

        # keep only species with >=20 observations
        species_counts = df["species_id"].value_counts()
        valid_species = species_counts[species_counts >= 1].index
        df = df[df["species_id"].isin(valid_species)].reset_index(drop=True)
        
        if label_map is None:
            unique_ids = sorted(df["species_id"].unique())
            self.label_map = {sid: i for i, sid in enumerate(unique_ids)}
        else:
            self.label_map = label_map
            
        # train / validation split
        df = df[df["subset"] == subset]

        print(df.columns)

        patches_dir = self.root / "patches"

        self.observations = []

        print("Building dataset...")

        for row in df.itertuples():

            obs_id = row.observation_id
            species = row.species_id
            lat = row.latitude
            lon = row.longitude

            region = "fr" 

            sub1 = str(obs_id)[-2:]
            sub2 = str(obs_id)[-4:-2]

            img_path = patches_dir / region / sub1 / sub2 / f"{obs_id}_rgb.jpg"

            if img_path.exists():
                self.observations.append((img_path, species, lat, lon))

        print("Images found:", len(self.observations))
        #print("Images found:", len(self.observations))

        self.labels = [self.label_map[s] for _, s, _, _ in self.observations]

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):

        img_path, species, lat, lon = self.observations[idx]

        obs_id = img_path.stem.replace("_rgb", "")
        nir_path = img_path.with_name(f"{obs_id}_near_ir.jpg")

        rgb = Image.open(img_path).convert("RGB")

        if nir_path.exists():
            nir = Image.open(nir_path).convert("L")
        else:
            nir = Image.new("L", rgb.size)

        rgb = np.array(rgb)
        nir = np.array(nir)

        nir = nir[:, :, None]

        image = np.concatenate([rgb, nir], axis=2)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255

        if self.transform:
            image = self.transform(image)

        # normalize coordinates
        coords = torch.tensor([
            lat/90,
            lon/180,
            np.sin(np.radians(lat)),
            np.cos(np.radians(lat)),
            np.sin(np.radians(lon)),
            np.cos(np.radians(lon))
        ], dtype=torch.float32)

        label = self.label_map[species]

        return image, label, coords
