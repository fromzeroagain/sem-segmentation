import torch, numpy as np, matplotlib.pyplot as plt, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.idd_dataset import IDDDataset, get_idd_loaders

ds = IDDDataset(
    config.IDD_IMAGES,
    config.IDD_MASKS,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
)

image, mask = ds[0]
print(
    f"image shape: {image.shape}  mask shape: {mask.shape}  mask unique: {torch.unique(mask).tolist()}"
)
assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0}), (
    "Mask has non-binary values"
)

tr, vl, te = get_idd_loaders(
    batch_size=config.BATCH_SIZE,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
)

imgs, masks = next(iter(tr))
print(f"batch images: {imgs.shape}  batch masks: {masks.shape}")
print("DAY 1 DONE")
