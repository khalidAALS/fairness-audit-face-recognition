import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T


DET_JSON = Path("outputs/detections_sample/detections.json")
OUT_NPY = Path("outputs/detections_sample/embeddings.npy")
OUT_META = Path("outputs/detections_sample/embeddings_meta.json")

IMG_SIZE = 160


def main():
    det = json.loads(DET_JSON.read_text())
    crops = [d for d in det if d.get("crop_path")]

    # Use MPS if available for embeddings (this usually works fine)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    embs = []
    meta = []

    with torch.no_grad():
        for i, d in enumerate(crops):
            img = Image.open(d["crop_path"]).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)  # [1,3,160,160]
            e = model(x).cpu().numpy().astype(np.float32)[0]  # [512]
            embs.append(e)
            meta.append({
                "crop_path": d["crop_path"],
                "gender": d["gender"],
                "race": d["race"],
                "age_group": d["age_group"],
                "prob": d["prob"],
            })
            print(f"[{i+1}/{len(crops)}] emb shape={e.shape}")

    embs = np.stack(embs, axis=0)
    np.save(OUT_NPY, embs)
    OUT_META.write_text(json.dumps(meta, indent=2))

    print("\nSaved embeddings:", OUT_NPY, "shape=", embs.shape)
    print("Saved meta:", OUT_META)
    print("Device used:", device)


if __name__ == "__main__":
    main()
