from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN

META = Path("data/meta/fairface_metadata.csv")
OUT_ROOT = Path("data/processed/fairface_aligned")
OUT_CSV = Path("data/processed/fairface_aligned/labels_subset.csv")

# Targets 
TRAIN_PER_GROUP = 1500   # race+gender groups => 7*2=14 groups => ~21k
VAL_PER_GROUP = 200      # ~2.8k

IMG_SIZE = 160

def sample_balanced(df, split, per_group):
    df = df[df["split"] == split].copy()
    # grouped by race+gender to balance fairness axes
    groups = []
    for (race, gender), g in df.groupby(["race", "gender"]):
        g = g.sample(n=min(per_group, len(g)), random_state=42)
        g["race_gender"] = f"{race}__{gender}"
        groups.append(g)
    return pd.concat(groups, ignore_index=True)

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "val").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(META)

    train_df = sample_balanced(df, "train", TRAIN_PER_GROUP)
    val_df = sample_balanced(df, "val", VAL_PER_GROUP)

    # MTCNN on CPU 
    mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, keep_all=False, post_process=True, device="cpu")

    out_rows = []

    def process_split(split_name, sdf):
        out_dir = OUT_ROOT / split_name
        ok = 0
        for i, row in sdf.reset_index(drop=True).iterrows():
            img = Image.open(row["image_path"]).convert("RGB")
            crop = mtcnn(img)
            if crop is None:
                continue

            # crop tensor [0,1] -> PIL
            crop_img = (crop.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            crop_pil = Image.fromarray(crop_img)

            fname = f"{split_name}_{i:06d}.jpg"
            out_path = out_dir / fname
            crop_pil.save(out_path, quality=95)

            out_rows.append({
                "aligned_path": str(out_path),
                "split": split_name,
                "gender": row["gender"],
                "race": row["race"],
                "age_group": row["age_group"],
            })
            ok += 1

            if (i + 1) % 200 == 0:
                print(f"{split_name}: processed {i+1}/{len(sdf)} ok={ok}")

        print(f"{split_name}: DONE ok={ok}/{len(sdf)}")

    print("Exporting TRAIN subset:", len(train_df))
    process_split("train", train_df)

    print("Exporting VAL subset:", len(val_df))
    process_split("val", val_df)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print("Saved labels:", OUT_CSV)
    print(out_df["split"].value_counts())
    print("Sample rows:\n", out_df.head())

if __name__ == "__main__":
    main()
