from datasets import FairFaceDataset
from torch.utils.data import DataLoader

dataset = FairFaceDataset(
    csv_file="data/meta/fairface_metadata.csv",
    split="train"
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

images, labels = next(iter(loader))

print("Batch image tensor shape:", images.shape)
print("Sample labels:", labels)
