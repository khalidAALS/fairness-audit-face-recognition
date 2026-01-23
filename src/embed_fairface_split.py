import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as T


META = Path("data/meta/fairface_metadata.csv")
MAPS = Path("data/meta/label_maps.json")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 160
BATCH = 32  # safe on 8GB

def load_maps():
    return json.loads(MAPS.read_text())


def encode_labels(df, maps):
    df = df.copy()
    df["gender_id"] = df["gender"].map(maps["gender"])
    df["race_id"] = df["race"].map(maps["race"])
    df["age_id"] = df["age_group"].map(maps["age_group"])
    return df


def main(split="train"):
    maps = load_maps()

    df = pd.read_csv(META)
    df = df[df["split"] == split].reset_index(drop=True)
    df = encode_labels(df, maps)

    # Alignment/detection on CPU 
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False, post_process=True, device="cpu")

    # Embeddings on MPS if available
    emb_device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = InceptionResnetV1(pretrained="vggface2").eval().to(emb_device)

    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    embs = np.zeros((len(df), 512), dtype=np.float32)
    ok = np.zeros((len(df),), dtype=np.uint8)

    batch_imgs = []
    batch_idx = []

    with torch.no_grad():
        for i, row in df.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")

            # aligned crop tensor (3,160,160) or None
            crop = mtcnn(img)
            if crop is None:
                # no face found
                continue

            # crop is already tensor in [0,1] when post_process=True
            # applies FaceNet normalisation
            crop_img = (crop * 255).byte().permute(1, 2, 0).numpy()
            crop_pil = Image.fromarray(crop_img)
            x = tfm(crop_pil)

            batch_imgs.append(x)
            batch_idx.append(i)

            if len(batch_imgs) == BATCH:
                X = torch.stack(batch_imgs, dim=0).to(emb_device)
                E = model(X).cpu().numpy().astype(np.float32)
                embs[batch_idx] = E
                ok[batch_idx] = 1
                batch_imgs, batch_idx = [], []

        # flushs last batch
        if batch_imgs:
            X = torch.stack(batch_imgs, dim=0).to(emb_device)
            E = model(X).cpu().numpy().astype(np.float32)
            embs[batch_idx] = E
            ok[batch_idx] = 1

    out_emb = OUT_DIR / f"fairface_embeddings_{split}.npy"
    out_lab = OUT_DIR / f"fairface_labels_{split}.csv"

    np.save(out_emb, embs)
    df_out = df[["image_path", "gender", "race", "age_group", "gender_id", "race_id", "age_id"]].copy()
    df_out["face_detected"] = ok
    df_out.to_csv(out_lab, index=False)

    print("Saved:", out_emb, "shape=", embs.shape)
    print("Saved:", out_lab)
    print("Detected faces:", int(ok.sum()), "/", len(ok))
    print("Embeddings device:", emb_device)


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(split=split)
