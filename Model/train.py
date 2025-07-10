import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import random
import time
import copy
from collections import defaultdict
from sklearn.model_selection import ShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import pandas as pd

# Dataset 
class BrainCTDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')  
        image = TF.to_tensor(image)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'mask': mask}

        return sample

# MODELO ATTENTION UNET
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

def init_weights(net, init_type='kaiming'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class UNet_Attention(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNet_Attention, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)

        return out

# Metricas
def dice_coeff_loss(pred=None, target=None, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dicecoeff = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return dicecoeff.mean(), loss.mean()

def focal_loss(alpha, gamma,ce_loss):
  pt = torch.exp(-ce_loss)
  focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
  return focal_loss

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bceweight = torch.ones_like (target)  +  20 * target
    bce = F.binary_cross_entropy_with_logits(pred, target, weight= bceweight)
    pred = torch.sigmoid(pred)
    dice_coeff, dice = dice_coeff_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return bce

def train_model(model, dataloaders, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    dice_coeff_dict = {'train':[],'validate':[]}
    bce_dict = {'train':[],'validate':[]}
    loss_dict = {'train':[],'validate':[]}
    phases = ['train','validate']
    best_loss = 1e10

    for epoch in range(num_epochs):
        since = time.time()
        for p in phases:
            if p == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0

            for data in dataloaders[p]:
                inputs = data['image'].to(device)
                labels = data['mask'].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = calc_loss(outputs, labels, metrics)
                if p == 'train':
                    loss.backward()
                    optimizer.step()
                epoch_samples += inputs.size(0)

            epoch_dice_coeff = metrics['dice_coeff']/ epoch_samples
            epoch_bce = metrics['bce']/epoch_samples
            epoch_loss = metrics['loss'] / epoch_samples
            dice_coeff_dict[p].append(epoch_dice_coeff)
            bce_dict[p].append(epoch_bce)
            loss_dict[p].append(epoch_loss)

            if p == 'validate' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            elif p == 'validate':
                if scheduler:
                    scheduler.step()
                    #scheduler.step(epoch_loss)

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, loss_dict, dice_coeff_dict, bce_dict

# Directorios con las imágenes y las máscaras
dataset_path = 'dataset_path'
train_image_dir = os.path.join(dataset_path, 'path_training_images')
train_mask_dir = os.path.join(dataset_path, 'path_training_masks')
image_paths = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir)])
mask_paths = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir)])

# Validación Cruzada
ss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
fold_metrics = []
best_dice = -1
best_model_state = None
best_fold = -1

for fold, (train_idx, val_idx) in enumerate(ss.split(image_paths)):
    print(f"\n Fold {fold+1}")

    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    train_dataset = BrainCTDataset(train_images, train_masks, augment=False)
    val_dataset = BrainCTDataset(val_images, val_masks, augment=False)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True),
        'validate': DataLoader(val_dataset, batch_size=8, shuffle=False)
    }

    model = UNet_Attention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.3)

    model_fold, loss_dict, dice_dict, bce_dict = train_model(model, dataloaders, optimizer, scheduler, num_epochs=60)

    # Extraer la última métrica de validación
    val_dice = dice_dict['validate'][-1]
    val_bce = bce_dict['validate'][-1]
    val_loss = loss_dict['validate'][-1]

    # Guardar métricas de este fold
    fold_metrics.append({
        'fold': fold + 1,
        'dice': val_dice,
        'bce': val_bce,
        'loss': val_loss
    })

    # Guardar modelo del fold
    torch.save(model_fold.state_dict(), f"modelo_fold_{fold+1}.pt")

    # Verificar si es el mejor modelo
    if val_dice > best_dice:
        best_dice = val_dice
        best_model_state = model_fold.state_dict()
        best_fold = fold + 1

# Promediar Métricas de Entrenamiento
torch.save(best_model_state, "mejor_modelo.pt")
print(f"Mejor modelo: fold {best_fold} con Dice = {best_dice:.4f}")

# Promedio y desviación estándar de métricas
dice_scores = [m['dice'] for m in fold_metrics]
bce_scores  = [m['bce'] for m in fold_metrics]
loss_scores = [m['loss'] for m in fold_metrics]

avg_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)
avg_bce = np.mean(bce_scores)
std_bce = np.std(bce_scores)
avg_loss = np.mean(loss_scores)
std_loss = np.std(loss_scores)

# Crear DataFrame a partir de fold_metrics
df = pd.DataFrame(fold_metrics)
promedios = df.mean()
desviaciones = df.std()
df.loc['Promedio'] = promedios
df.loc['Desviación Estándar'] = desviaciones
# Guardar en CSV
df.to_csv("metricas_validacion_cruzada.csv", float_format='%.4f')
