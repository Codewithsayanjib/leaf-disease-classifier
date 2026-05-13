
import torch
DATA_DIR = "data/PlantVillage/PlantVillage"
CHECKPOINT_DIR = "outputs/checkpoints"
PLOTS_DIR      = "outputs/plots"


IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 15
LR          = 1e-4
NUM_WORKERS = 4
SEED        = 42


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")