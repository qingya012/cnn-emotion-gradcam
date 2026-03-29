import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/fer2013.csv")

print(df.head())
print(df.columns)

row = df.iloc[0]

label = row["emotion"]
pixels = row["pixels"].split()
pixels = np.array(pixels, dtype=np.float32)
img = pixels.reshape(48, 48)

print("Label: ", label)
print("Image shape: ", img.shape)

plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.show()