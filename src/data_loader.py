import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

class FERDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        label = int(row["emotion"])

        pixels = row["pixels"].split()
        pixels = np.array(pixels, dtype=np.float32).reshape(48, 48)

        img = torch.tensor(pixels).unsqueeze(0)

        return img, label

if __name__ == "__main__":
    dataset = FERDataset("data/fer2013.csv")
    print("Dataset size: ", len(dataset))

    img, label = dataset[0]
    print("Single sample: ", img.shape, label)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(loader))
    print("Batch shape: ", images.shape, labels.shape)
