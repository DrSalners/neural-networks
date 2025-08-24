import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

data=datasets.MNIST(root='./data',
                    download=True,
                    train=True,
                    transform=ToTensor())

# plt.imshow(data.__getitem__(0)[0].detach().permute(1,2,0).numpy())
# plt.show()

batch_size=1
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

for x_batch, y_batch in data_loader:
    print(x_batch,y_batch)
    break
