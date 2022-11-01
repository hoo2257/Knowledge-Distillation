import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from networks import Teacher
from utils import createFolder, train_val
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import os
import copy
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load MNIST dataset
createFolder('./data')
# Define transformation
ds_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST('/content/data', train=True, download=True, transform=ds_transform)
val_ds = datasets.MNIST('/content/data', train=False, download=True, transform=ds_transform)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=True)

for x,y in train_dl:
    print(x.shape, y.shape)
    break

num = 4
img = x[:num]

plt.figure(figsize=(15,15))
for i in range(num):
    plt.subplot(1, num+1, i+1)
    plt.imshow(to_pil_image(0.1307*img[i]+0.3081), cmap='gray')


# Check Teacher Model
print(device)
x = torch.randn(16,1,28,28).to(device)
teacher = Teacher().to(device)
output = teacher(x)
print(output.shape)

# Train Teacher Model
loss_func = nn.CrossEntropyLoss() # loss function

# optimizer
opt = optim.Adam(teacher.parameters())

# lr scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

params_train = {
    'num_epochs':30,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/teacher_weights.pt',
    'device':device,
}

createFolder('./models')

teacher, loss_hist, metric_hist = train_val(teacher, params_train)









