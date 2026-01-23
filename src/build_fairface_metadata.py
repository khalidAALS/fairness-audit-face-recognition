import pandas as pd
from pathlib import Path

ROOT = Path("data/raw/fairface")
OUT = Path("data/meta/fairface_metadata.csv")

def load_split(csv_file, split_name):
    df = pd.read_csv(csv_file)
    df["split"] = split_name
    df["image_path"] = df["file"].apply(lambda x: f"data/raw/fairface/images/{x}")
    return df

def main():
    train_csv = ROOT / "fairface_label_train.csv"
    val_csv = ROOT / "fairface_label_val.csv"

    train_df = load_split(train_csv, "train")
    val_df = load_split(val_csv, "val")

    df = pd.concat([train_df, val_df], ignore_index=True)

    # Standard schema
    df = df.rename(columns={
        "gender": "gender",
        "race": "race",
        "age": "age_group"
    })

    df = df[["image_path", "gender", "race", "age_group", "split"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("Metadata created:", OUT)
    print(df.head())
    print("\nCounts by subgroup:")
    print(df.groupby(["race", "gender"]).size())

if __name__ == "__main__":
    main()
