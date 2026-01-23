import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as T
import torchvision.models as models


META = Path("data/meta/fairface_metadata.csv")
MAPS = Path("data/meta/label_maps.json")
CKPT = Path("models/arcface_gender_resnet18.pth")

OUT_EMB = Path("outputs/arcface_embeddings_train.npy")
OUT_LAB = Path("outputs/arcface_labels_train.csv")

IMG_SIZE = 160
BATCH = 64


class Backbone(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.proj = nn.Linear(in_features, emb_dim)

    def forward(self, x):
        feat = self.base(x)
        emb = self.proj(feat)
        return F.normalize(emb)


def main():
    maps = json.loads(MAPS.read_text())
    df = pd.read_csv(META)
    df = df[df["split"] == "train"].reset_index(drop=True)

    df["gender_id"] = df["gender"].map(maps["gender"])
    df["race_id"] = df["race"].map(maps["race"])
    df["age_id"] = df["age_group"].map(maps["age_group"])

    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False, post_process=True, device="cpu")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = Backbone(emb_dim=256).to(device).eval()

    ckpt = torch.load(CKPT, map_location="cpu")
    backbone.load_state_dict(ckpt["backbone"], strict=True)

    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    embs = np.zeros((len(df), 256), dtype=np.float32)
    ok = np.zeros((len(df),), dtype=np.uint8)

    batch = []
    idxs = []

    with torch.no_grad():
        for i, row in df.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")
            crop = mtcnn(img)
            if crop is None:
                continue

            crop_img = (crop.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            crop_pil = Image.fromarray(crop_img)
            x = tfm(crop_pil)

            batch.append(x)
            idxs.append(i)

            if len(batch) == BATCH:
                X = torch.stack(batch).to(device)
                E = backbone(X).cpu().numpy().astype(np.float32)
                embs[idxs] = E
                ok[idxs] = 1
                batch, idxs = [], []

        if batch:
            X = torch.stack(batch).to(device)
            E = backbone(X).cpu().numpy().astype(np.float32)
            embs[idxs] = E
            ok[idxs] = 1

    np.save(OUT_EMB, embs)

    out_df = df[["image_path", "gender", "race", "age_group", "gender_id", "race_id", "age_id"]].copy()
    out_df["face_detected"] = ok
    out_df.to_csv(OUT_LAB, index=False)

    print("Saved:", OUT_EMB, "shape=", embs.shape)
    print("Saved:", OUT_LAB)
    print("Detected faces:", int(ok.sum()), "/", len(ok))
    print("Device:", device)


if __name__ == "__main__":
    main()
