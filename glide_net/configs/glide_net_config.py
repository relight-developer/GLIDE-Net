"""
Default configuration for GLIDE-Net.
"""
import torch


class GLIDE_NetConfig:
    """Hyper-parameters used by the training and evaluation scripts."""

    # Training
    BATCH_SIZE = 12
    LEARNING_RATE = 1e-3
    MIN_LEARNING_RATE = 1e-7
    WEIGHT_DECAY = 5e-4
    NUM_EPOCHS = 50

    # Input
    INPUT_SIZE = 352

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
