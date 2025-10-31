import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from datasets.nets_utility import *

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class MFIF_dataset_finetune_aug(Dataset):
    def __init__(self, base_dir=r'G:\project\Fuse_any_focus\get_MFIF_dataset\MFIF-real-basic', split="train"):
        self.train_img_dir_A = base_dir + '/train/image_A'
        self.train_img_dir_B = base_dir + '/train/image_B'
        self.train_img_dir_C = base_dir + '/train/image_C'
        self.train_label_dir = base_dir + '/train/label'
        self.test_img_dir_A = base_dir + '/test/image_A'
        self.test_img_dir_B = base_dir + '/test/image_B'
        self.test_label_dir = base_dir + '/test/label'
        self.train_list = os.listdir(self.train_img_dir_A)
        try:
            self.test_list = os.listdir(self.test_img_dir_A)
        except:
            print('no test dir')
        self.split = split
        self.random_erasing = True
        self.random_offset = True
        self.gaussian_noise = True
        self.random_colorjitter = True
        print('used augmentation')


    def __len__(self):
        return len(self.train_list)

    def augment(self, image_A, image_B, label):
        if self.random_colorjitter:
            if np.random.rand() > 0.99:
                image_A = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image_A)
            if np.random.rand() > 0.99:
                image_B = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image_B)
            if np.random.rand() > 0.99:
                image_A = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image_A)
                image_B = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)(image_B)
        if self.random_erasing:
            if np.random.rand() > 0.99:
                image_A, image_B = random_erasing(image_A, image_B, 6, 15, 20)
        # random offset
        if self.random_offset:
            image_A, image_B = random_offset(image_A, image_B, 2, 2)
        # gaussian noise
        if self.gaussian_noise:
            std = torch.rand(1) * 0.1
            image_A, image_B = gaussian_noise(image_A, image_B, std)
        return image_A, image_B, label

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.train_list[idx].split('.')[0]
            image_path_A = os.path.join(self.train_img_dir_A, slice_name+'.jpg')
            image_path_B = os.path.join(self.train_img_dir_B, slice_name+'.jpg')
            image_path_C = os.path.join(self.train_img_dir_C, slice_name+'.jpg')
            # use torchvision to read the jpg
            try:
                image_A = cv2.imread(image_path_A)
                image_B = cv2.imread(image_path_B)
                image_C = transforms.ToTensor()(cv2.imread(image_path_C))
                image = torch.cat([transforms.ToTensor()(image_A), transforms.ToTensor()(image_B)], dim=0)
                label_path = os.path.join(self.train_label_dir, slice_name + '.pt')
                label = torch.load(label_path, weights_only=True)
            except:
                print(image_path_A, image_path_B)
                # delete the image
                # os.remove(image_path_A)
                # os.remove(image_path_B)
                # os.remove(os.path.join(self.train_label_dir, slice_name+'.pt'))
                image = None
                label = None
        else:
            slice_name = self.test_list[idx].split('.')[0]
            image_path_A = os.path.join(self.test_img_dir_A, slice_name+'.jpg')
            image_path_B = os.path.join(self.test_img_dir_B, slice_name+'.jpg')
            image_path_C = os.path.join(self.test_img_dir_C, slice_name+'.jpg')
            # use torchvision to read the jpg
            try:
                image_A = cv2.imread(image_path_A)
                image_B = cv2.imread(image_path_B)
                image_C = transforms.ToTensor()(cv2.imread(image_path_C))
                image = torch.cat([transforms.ToTensor()(image_A), transforms.ToTensor()(image_B)], dim=0)
            except:
                print(image_path_A, image_path_B)
                # delete the image
                # os.remove(image_path_A)
                # os.remove(image_path_B)
                # os.remove(os.path.join(self.train_label_dir, slice_name + '.pt'))
            label_path = os.path.join(self.train_label_dir, slice_name + '.pt')
            label = torch.load(label_path)  # 512*512
            # label = torch.from_numpy(data).unsqueeze(0)


        image_A, image_B, label = self.augment(transforms.ToTensor()(image_A).unsqueeze(0), transforms.ToTensor()(image_B).unsqueeze(0), label)
        image = torch.cat([image_A[0], image_B[0]], dim=0)
        # sample = {'image': image, 'label': label}
        sample = {'image_A': image_A[0], 'image_B': image_B[0], 'image': image, 'label': label, 'image_C': image_C}
        sample['case_name'] = self.train_list[idx].split('.')[0]
        return sample

if __name__ == '__main__':
    dataset = MFIF_dataset_finetune_aug()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        try:
            print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
        except:
            continue