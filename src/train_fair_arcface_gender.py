import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models


LABELS = Path("data/processed/fairface_aligned/labels_subset.csv")
MAPS = Path("data/meta/label_maps.json")
OUT = Path("models/fair_arcface_gender_resnet18.pth")

IMG_SIZE = 160
BATCH = 64
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4


class AlignedCSV(Dataset):
    def __init__(self, df, maps, split="train"):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.maps = maps
        self.tfm = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["aligned_path"]).convert("RGB")
        x = self.tfm(img)
        y = self.maps["gender"][r["gender"]]  # gender task
        return x, int(y)


class ArcMarginProduct(nn.Module):
    """ArcFace-style margin layer."""
    def __init__(self, in_features, out_features, s=20.0, m=0.20):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        x = F.normalize(x)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cosine)
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        out = cosine * (1.0 - one_hot) + target * one_hot
        return out * self.s


class Backbone(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
    df = pd.read_csv(LABELS)

    train_ds = AlignedCSV(df, maps, split="train")
    val_ds = AlignedCSV(df, maps, split="val")

    # Balanced sampler (prevents single-class collapse + stabilizes fairness)
    labels = np.array([train_ds[i][1] for i in range(len(train_ds))])
    class_counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = Backbone(emb_dim=256).to(device)
    head = ArcMarginProduct(in_features=256, out_features=2, s=20.0, m=0.20).to(device)

    opt = torch.optim.AdamW(list(backbone.parameters()) + list(head.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    def eval_acc():
        backbone.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                emb = backbone(x)
                logits = head(emb, y)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / max(1, total)

    for epoch in range(1, EPOCHS + 1):
        backbone.train()
        head.train()
        running = 0.0

        for step, (x, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            emb = backbone(x)
            logits = head(emb, y)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {running/50:.4f}")
                running = 0.0

        acc = eval_acc()
        print(f"epoch {epoch} VAL acc: {acc:.4f}")

    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()}, OUT)
    print("Saved:", OUT)
    print("Device:", device)
    print("Train class counts (0=Female,1=Male):", class_counts.tolist())


if __name__ == "__main__":
    main()
