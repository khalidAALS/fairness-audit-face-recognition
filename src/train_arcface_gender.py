import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models


LABELS = Path("data/processed/fairface_aligned/labels_subset.csv")
MAPS = Path("data/meta/label_maps.json")
OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 160
BATCH = 64
EPOCHS = 3
LR = 1e-3

# train for gender classification first (2 classes)
TASK = "gender"  # can add race/age


class AlignedSubset(Dataset):
    def __init__(self, df, maps, task="gender", split="train"):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.maps = maps
        self.task = task

        self.tfm = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(r["aligned_path"]).convert("RGB")
        x = self.tfm(img)

        if self.task == "gender":
            y = self.maps["gender"][r["gender"]]
        elif self.task == "race":
            y = self.maps["race"][r["race"]]
        elif self.task == "age_group":
            y = self.maps["age_group"][r["age_group"]]
        else:
            raise ValueError("unknown task")

        meta = {"race": r["race"], "gender": r["gender"], "age_group": r["age_group"]}
        return x, int(y), meta


class ArcMarginProduct(nn.Module):
    """ArcFace-style margin layer."""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        # normalizes
        x = F.normalize(x)
        W = F.normalize(self.weight)

        cosine = F.linear(x, W)  # [B, C]
        # clamps for numerical stability
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = cosine * (1.0 - one_hot) + target_logit * one_hot
        output *= self.s
        return output


class Backbone(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        base = models.resnet18(weights=None)
        # replaces final layer
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.base = base
        self.proj = nn.Linear(in_features, emb_dim)

    def forward(self, x):
        feat = self.base(x)
        emb = self.proj(feat)
        emb = F.normalize(emb)
        return emb


def main():
    maps = json.loads(MAPS.read_text())
    df = pd.read_csv(LABELS)

    n_classes = len(maps["gender"]) if TASK == "gender" else len(maps[TASK])

    train_ds = AlignedSubset(df, maps, task=TASK, split="train")
    val_ds = AlignedSubset(df, maps, task=TASK, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = Backbone(emb_dim=256).to(device)
    head = ArcMarginProduct(in_features=256, out_features=n_classes, s=30.0, m=0.50).to(device)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=LR)

    def eval_acc():
        backbone.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                emb = backbone(x)
                logits = head(emb, y)  # uses y just for forward, for eval ok
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        return correct / max(1, total)

    for epoch in range(1, EPOCHS + 1):
        backbone.train()
        head.train()
        running = 0.0
        for step, (x, y, _) in enumerate(train_loader, 1):
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

    # Save model
    out = OUT_DIR / "arcface_gender_resnet18.pth"
    torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()}, out)
    print("Saved:", out)
    print("Device:", device)


if __name__ == "__main__":
    main()
