import os, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class IDDDataset(Dataset):
    def __init__(
        self, image_dir, mask_dir, img_height=512, img_width=512, augment=False
    ):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.img_height, self.img_width = img_height, img_width
        self.pairs = self._build_pairs()
        self.augment = augment
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD),
            ]
        )

    def _build_pairs(self):
        images = sorted(os.listdir(self.image_dir))
        masks = sorted(os.listdir(self.mask_dir))

        assert len(images) == len(masks), (
            f"Mismatch: {len(images)} images vs {len(masks)} masks."
        )

        pairs = [
            (os.path.join(self.image_dir, i), os.path.join(self.mask_dir, m))
            for i, m in zip(images, masks)
        ]

        print(f"IDDDataset:{len(pairs)} pairs")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(
            mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST
        )

        binary = (mask == 0) if config.MASK_INVERTED else (mask > 0)
        binary = binary.astype(np.float32)

        if self.augment and np.random.random() > 0.5:
            img, binary = cv2.flip(img, 1), cv2.flip(binary, 1)

        # C.H,W :0,1,2
        # H,W,C :1,2,0
        image_tensor = self.normalize(img)
        mask_tensor = torch.from_numpy(binary).unsqueeze(0)
        return image_tensor, mask_tensor


def get_idd_loaders(batch_size=8, **kw):
    from torch.utils.data import SubsetRandomSampler

    full = IDDDataset(config.IDD_IMAGES, config.IDD_MASKS, **kw)
    idx = list(range(len(full)))
    train_idx, temp_idx = train_test_split(idx, test_size=0.25, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.4, random_state=42)

    print(f"Split train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    train_ds = IDDDataset(config.IDD_IMAGES, config.IDD_MASKS, augment=True, **kw)
    eval_ds = IDDDataset(config.IDD_IMAGES, config.IDD_MASKS, augment=False, **kw)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            drop_last=True,
        ),
        DataLoader(
            eval_ds, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx)
        ),
        DataLoader(
            eval_ds, batch_size=batch_size, sampler=SubsetRandomSampler(test_idx)
        ),
    )
