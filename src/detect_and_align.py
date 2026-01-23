import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN


META = Path("data/meta/fairface_metadata.csv")
OUT_DIR = Path("outputs/detections_sample")
CROPS_DIR = OUT_DIR / "crops"
OUT_JSON = OUT_DIR / "detections.json"

SAMPLE_N = 20
IMG_SIZE = 160  # FaceNet style crop size


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META).sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)

    # Uses MPS if available (faster), else CPU
    device = "cpu"
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False, post_process=False, device=device)

    results = []
    for i, row in df.iterrows():
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")

        boxes, probs = mtcnn.detect(img)

        rec = {
            "index": int(i),
            "image_path": img_path,
            "gender": row["gender"],
            "race": row["race"],
            "age_group": row["age_group"],
            "detected": bool(boxes is not None and len(boxes) > 0),
            "box": None,
            "prob": None,
            "crop_path": None,
        }

        if rec["detected"]:
            rec["box"] = [float(x) for x in boxes[0].tolist()]
            rec["prob"] = float(probs[0]) if probs is not None else None

            crop = mtcnn(img)  # tensor (3, IMG_SIZE, IMG_SIZE)
            if crop is not None:
                crop = crop.clamp(0, 255).byte()  # because post_process=False gives 0..255
                crop_img = crop.permute(1, 2, 0).cpu().numpy()
                crop_pil = Image.fromarray(crop_img)
                crop_path = CROPS_DIR / f"crop_{i:03d}.jpg"
                crop_pil.save(crop_path, quality=95)
                rec["crop_path"] = str(crop_path)

        results.append(rec)
        print(f"[{i+1}/{SAMPLE_N}] detected={rec['detected']} prob={rec['prob']} crop={rec['crop_path']}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print("\nSaved:", OUT_JSON)
    print("Crops in:", CROPS_DIR)


if __name__ == "__main__":
    main()
