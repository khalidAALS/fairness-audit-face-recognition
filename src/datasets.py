import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FairFaceDataset(Dataset):
    def __init__(self, csv_file, split="train", transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = {
            "gender": row["gender"],
            "race": row["race"],
            "age_group": row["age_group"]
        }

        return img, label
