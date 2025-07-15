from os import environ
import torch

# for reproducability
SEED: int = 87

# force everything to the cpu
environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")
