import json
import pandas as pd
from pathlib import Path

META = Path("data/meta/fairface_metadata.csv")
OUT = Path("data/meta/label_maps.json")

def main():
    df = pd.read_csv(META)

    maps = {}
    for col in ["gender", "race", "age_group"]:
        classes = sorted(df[col].dropna().unique().tolist())
        maps[col] = {cls: i for i, cls in enumerate(classes)}

    OUT.write_text(json.dumps(maps, indent=2))
    print("Saved:", OUT)
    for k, v in maps.items():
        print(k, "classes:", len(v))
        print(v)

if __name__ == "__main__":
    main()
