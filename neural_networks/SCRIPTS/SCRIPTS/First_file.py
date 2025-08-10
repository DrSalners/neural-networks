
import torch
import torchvision
import torchaudio

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)

print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
