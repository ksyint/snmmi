import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import csv
import numpy as np
import os
import pandas as pd
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchsummary import summary
from natsort import natsorted
import pydicom
from glob import glob
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk
import tqdm
from functools import lru_cache
from importlib import reload
from utils.tools import CLASSES, COLORDICT
writer = SummaryWriter()


class GaainDataset(BaseDataset):
    CLASSES = CLASSES
    COLORDICT = COLORDICT

    def __init__(self, patient_dir, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
        self.ids = sorted(os.listdir(patient_dir))
        self.images_fps = [os.path.join(patient_dir, image_id, images_dir) for image_id in self.ids]
        self.masks_fps = [os.path.join(patient_dir, image_id, masks_dir) for image_id in self.ids]

        # images와 masks 파일이 다 있는지 확인 검증
        is_del = False
        for idx, (img, msk) in enumerate(zip(self.images_fps, self.masks_fps)):
            if not os.path.exists(img):
                is_del = True
            if not os.path.exists(msk):
                is_del = True
            if is_del:
                print(f'Removed {self.ids[idx]}')
                del self.ids[idx]
                del self.images_fps[idx]
                del self.masks_fps[idx]
                is_del = False

        # convert str names to class values on masks
        self.class_values = None if classes == None else list(self.CLASSES.keys()) if classes == [] else [key for
                                                                                                          key, val in
                                                                                                          self.CLASSES.items()
                                                                                                          if
                                                                                                          val.lower() in [
                                                                                                              el.lower()
                                                                                                              for el in
                                                                                                              classes]]
        if classes is not None:
            if 0 in classes:
                self.class_values.remove(0)
        # print(self.class_values, classes, None if classes == None else 'hi')
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # print(f'PID: {self.ids[idx]}')
        image_path = self.images_fps[idx]
        mask_path = self.masks_fps[idx]
        itk_image = sitk.ReadImage(image_path)
        itk_mask = sitk.ReadImage(mask_path)
        numpy_image = sitk.GetArrayFromImage(itk_image).astype('float32')  # Convert the image to a NumPy array
        numpy_mask = sitk.GetArrayFromImage(itk_mask).astype('int64')  # Convert the image to a NumPy array
        torch_image = torch.from_numpy(numpy_image)  # Convert the NumPy array to a Torch tensor
        torch_mask = torch.from_numpy(numpy_mask)  # Convert the NumPy array to a Torch tensor

        if self.class_values == None:
            raise Exception("ERROR: classes is None")
        elif self.class_values == []:
            pass
            # self.class_values = list(torch.unique(torch_mask))
            # self.class_values = list(self.CLASSES.keys()).remove(0)

            # boolean_masks = [(torch_mask == v) for v in self.class_values] # Convert the NumPy array to a Torch tensor
            # sum_mask = torch.stack(boolean_masks).sum(dim=0).clamp(max=1).int()
            # torch_mask *= sum_mask
        else:
            boolean_masks = [(torch_mask == v) for v in self.class_values]  # Convert the NumPy array to a Torch tensor
            sum_mask = torch.stack(boolean_masks).sum(dim=0).clamp(max=1).int()
            torch_mask *= sum_mask

        one_hot_masks, _ = self.one_hot_encode(torch_mask)

        # Optionally, you may add some pre-processing or data augmentation here.
        if self.augmentation:
            subject_dict = {
                'volume': tio.ScalarImage(tensor=torch_image.unsqueeze(0)),
                'mask': tio.LabelMap(tensor=torch_mask.unsqueeze(0)),
            }

            for i in range(one_hot_masks.shape[0]):
                key = f'one_hot_label_{i}'
                value = one_hot_masks[i].unsqueeze(0)
                subject_dict[key] = tio.LabelMap(tensor=value)

            subject = tio.Subject(**subject_dict)
            transformed_subject = self.augmentation(subject)
            torch_image, torch_mask = transformed_subject['volume'].data.squeeze(0), transformed_subject[
                'mask'].data.squeeze(0)

            tmp_masks = []
            for i in range(one_hot_masks.shape[0]):  # 11은 one_hot_mask의 channel 수를 나타냅니다.
                key = f'one_hot_label_{i}'
                one_hot_channel = transformed_subject[key].data.squeeze(0)
                tmp_masks.append(one_hot_channel)

            # one_hot_masks를 하나의 tensor로 합치기
            one_hot_masks = torch.stack(tmp_masks, dim=0)  # 결과: 11 x 64 x 64 x 64

        if self.preprocessing:
            pass
            # sample = self.preprocessing(image=torch_image, mask=torch_mask)
            # torch_image, torch_mask = sample['Image'], sample['mask']

        return torch_image.unsqueeze(0), torch_mask.unsqueeze(0), one_hot_masks.unsqueeze(1)

    def one_hot_encode(self, mask):

        unique_vals = torch.tensor(self.class_values)

        # 0 값을 제외
        unique_vals = unique_vals[unique_vals != 0]

        one_hot = torch.zeros((len(unique_vals), *mask.shape), dtype=torch.float32)

        one_hot_map = {}
        for idx, val in enumerate(unique_vals):
            one_channel = (mask == val).to(torch.float32)
            one_hot[idx] = one_channel
            one_hot_map[idx] = [val.item(), self.CLASSES[val.item()]]

        return one_hot, one_hot_map

    @lru_cache(maxsize=1)  # 무제한 캐시 크기; 필요에 따라 maxsize를 설정할 수 있습니다.
    def freesurfer_colormap(self, cmap):
        if cmap == 'lut':
            colordict = self.COLORDICT
        # 0부터 2035까지의 모든 값에 대해 RGB 값을 매핑
        rgb_values = [(0, 0, 0) for _ in range(2036)]

        for key, value in colordict.items():
            if key < 2036:  # 이 부분은 주어진 범위 내에 있는 키만 처리하기 위함
                rgb_values[key] = value

        # RGB 값들을 0~1 범위로 정규화
        normalized_rgb_values = [(r / 255, g / 255, b / 255) for r, g, b in rgb_values]

        # 커스텀 colormap 생성
        custom_colormap = mcolors.ListedColormap(normalized_rgb_values)

        return custom_colormap

    def display_mpr(self, volume, cmap='gray'):
        def pad(img, target_size):
            pad = np.array(target_size) - np.array(img.shape)
            img = np.pad(img, [(pad[0] // 2, pad[0] - pad[0] // 2), (pad[1] // 2, pad[1] - pad[1] // 2)])
            return img

        c = np.array(volume.shape) // 2
        ax = volume[c[0], :, :]
        co = volume[:, c[1], :]
        sa = volume[:, :, c[2] + c[2] // 3]

        h = np.max([ax.shape, co.shape, sa.shape], axis=0)[0]
        ax = pad(ax, [h, ax.shape[1]])
        co = pad(co, [h, co.shape[1]])
        sa = pad(sa, [h, sa.shape[1]])

        dis = np.hstack([ax, co, sa])

        fig, ax = plt.subplots(figsize=(15, 15))
        if cmap == 'lut':
            cmap = self.freesurfer_colormap(cmap)
            im = ax.imshow(dis, cmap=cmap, interpolation='nearest', vmin=0, vmax=2035)
        else:
            im = ax.imshow(dis, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()

resolution = 64
training_transform = tio.Compose([
    tio.Resize(resolution),
    # tio.ToCanonical(),
    # tio.RandomElasticDeformation(
    #     num_control_points=(5, 5, 5),  # 제어점의 수를 정의
    #     max_displacement=(10, 10, 10),  # 최대 변위량을 정의
    #     p=0.3  # 20% 확률로 적용
    # ),
    # tio.RandomBiasField(p=0.3),
    # # tio.RandomNoise(p=0.3),
    # # tio.RandomFlip(axes=(0,1,2),p=0.5),
    # tio.RandomAffine(
    #     scales=(0.9, 1.1),
    #     degrees=(15, 15, 15),
    #     translation=(10, 10, 10),
    #     p=0.3
    # ),
    tio.ZNormalization(),
])
testing_transform = tio.Compose([
    tio.Resize(resolution),
    # tio.ToCanonical(),
    # tio.RandomElasticDeformation(
    #     num_control_points=(5, 5, 5),  # 제어점의 수를 정의
    #     max_displacement=(10, 10, 10),  # 최대 변위량을 정의
    #     p=0.3  # 20% 확률로 적용
    # ),
    # # tio.RandomBiasField(p=0.3),
    # # tio.RandomNoise(p=0.3),
    # # tio.RandomFlip(axes=(0,1,2),p=0.5),
    # tio.RandomAffine(
    #     scales=(0.9, 1.1),
    #     degrees=(15, 15, 15),
    #     translation=(10, 10, 10),
    #     p=0.3
    # ),
    tio.ZNormalization(),
])

skull_striping_dict = {
    'train_dir': os.path.join('data','FREESURFER_MRI_NII'),
    'test_dir': os.path.join('data','FREESURFER_MRI_TEST'),
    'orig_file': 'orig.nii.gz',
    'mask_file': 'brainmask_binary.nii.gz',
    'classes': [
        'brain',
    ],
}
aseg_dict = {
    'train_dir': os.path.join('data','FREESURFER_MRI_NII'),
    'test_dir': os.path.join('data','FREESURFER_MRI_TEST'),
    'orig_file': 'orig.nii.gz',
    'mask_file': 'aseg.nii.gz',
    'classes': [
        # 'Left-Cerebral-White-Matter',
        # 'Left-Cerebral-Cortex',
        # 'Left-Cerebellum-White-Matter',
        # 'Left-Cerebellum-Cortex',
        'Left-Caudate',
        'Left-Putamen',
        # 'Left-Pallidum',
        # 'Left-Hippocampus',
        # 'Left-Amygdala',

        # 'Right-Cerebral-White-Matter',
        # 'Right-Cerebral-Cortex'
        # 'Right-Cerebellum-White-Matter',
        # 'Right-Cerebellum-Cortex',
        'Right-Caudate',
        'Right-Putamen',
        # 'Right-Pallidum',
        # 'Right-Hippocampus',
        # 'Right-Amygdala',
    ]
}
aparc_dict = {
    'train_dir': os.path.join('data','FREESURFER_MRI_NII'),
    'test_dir': os.path.join('data','FREESURFER_MRI_TEST'),
    'orig_file': 'orig.nii.gz',
    'mask_file': 'aparc+aseg.nii.gz',
    'classes': [
        'Left-Cerebral-White-Matter',
        # 'Left-Cerebral-Cortex',
        # 'Left-Cerebellum-White-Matter',
        # 'Left-Cerebellum-Cortex',
        # 'Left-Caudate',
        # 'Left-Putamen',
        # 'Left-Pallidum',
        # 'Left-Hippocampus',
        # 'Left-Amygdala',

        'Right-Cerebral-White-Matter',
        # 'Right-Cerebral-Cortex'
        # 'Right-Cerebellum-White-Matter',
        # 'Right-Cerebellum-Cortex',
        # 'Right-Caudate',
        # 'Right-Putamen',
        # 'Right-Pallidum',
        # 'Right-Hippocampus',
        # 'Right-Amygdala',
    ]
}

_dict = skull_striping_dict

train_dir, test_dir, orig_file, mask_file, classes = _dict.values()
dset = GaainDataset(train_dir, orig_file, mask_file, classes=classes,
                    augmentation=training_transform
                   )
tset = GaainDataset(test_dir, orig_file, mask_file, classes=classes,
                    augmentation=testing_transform
                   )
train_loader = DataLoader(dset, batch_size=2, shuffle=True)
test_loader = DataLoader(tset, batch_size=2, shuffle=False)

import torch.nn.functional as F


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        input = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs_sum = inputs.sum(dim=tuple(range(1,inputs.dim())))
        targets_sum = targets.sum(dim=tuple(range(1,targets.dim())))

        intersection = (inputs * targets).sum(dim=tuple(range(1,targets.dim())))
        dice = (2 * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        return (1 - dice).mean()

import models.unet as unet
reload(unet)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = unet.UNet3D(in_channels=1, num_classes=dset.class_values.__len__()).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = DiceLoss()
loss_fn2 = DiceLoss2()

EPOCH = 10
losses = []
for epoch in range(0, EPOCH):
    model.train()
    train_iterator = tqdm.tqdm(train_loader)
    sum_loss = 0
    for batch_idx, (data, _, one_hot_masks) in enumerate(train_iterator):
        opt.zero_grad()
        data = data.to(device)
        # print('before1:', data.shape, one_hot_masks.shape)
        out_msk = model(data)
        one_hot_masks = one_hot_masks.squeeze(2).to(device)
        # print('after:', out_msk.shape, one_hot_masks.shape)
        loss = loss_fn(out_msk, one_hot_masks, 1)
        loss2 = loss_fn2(out_msk, one_hot_masks, 1)
        print(f"**********{loss}, {loss2}********")
        loss.backward()
        opt.step()
        sum_loss += loss.item()
    mean_loss = sum_loss / len(train_loader)
    writer.add_scalar("Loss/train", mean_loss, epoch)
    print(f"Train Epoch {epoch} / {EPOCH} | Mean mean_loss {mean_loss:.03f}")
    # losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_iterator = tqdm.tqdm(test_loader)
        sum_loss = 0
        for batch_idx, (data, _, one_hot_masks) in enumerate(test_iterator):
            data = data.to(device)
            one_hot_masks = one_hot_masks.squeeze(2).to(device)
            out_msk = model(data)
            loss = loss_fn(out_msk, one_hot_masks, 1)
            sum_loss += loss.item()
        mean_loss = sum_loss / len(test_loader)
        writer.add_scalar("Loss/test", mean_loss, epoch)
        print(f"Test Epoch {epoch + 1} / {EPOCH} | Mean mean_loss {mean_loss:.03f}")

    # out_msk[0,0,:,:]
    # img_grid = make_grid(images)
    # writer.add_image('epoch_{}_images'.format(epoch), img_grid)
    writer.flush()

writer.close()