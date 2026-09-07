import os, torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
IDD_IMAGES = os.path.join(DATASETS_DIR, "idd", "image_archive")
IDD_MASKS = os.path.join(DATASETS_DIR, "idd", "mask_archive")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_HEIGHT, IMG_WIDTH = 512, 512
BATCH_SIZE = 8
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

MASK_INVERTED = False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
