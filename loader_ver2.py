import torch
from torch.utils.data import Dataset
from os.path import splitext
from os import listdir
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from utils import load_nii
from sklearn.model_selection import KFold
from collections import OrderedDict
import csv
import pandas as pd


def load_rupture_ids(rupture_dir, prefix=False):
    if rupture_dir is None:
        rupture_dir = '/home/xxx'
    filenames = listdir(rupture_dir)
    rupture_file = {}
    rupture_file[1] = []
    rupture_file[0] = []
    for filename in filenames:
        tmp_name = filename
        is_rupture = 1 if tmp_name[0] == 'r' else 0
        pat_name = tmp_name[2: -4] if prefix is False else tmp_name[: -4]
        if is_rupture == 1:
            rupture_file[1].append(pat_name)
        if is_rupture == 0:
            rupture_file[0].append(pat_name)
    return rupture_file


def split_data(patient_list, K=5, shuffle=False):
    """
    :param patient_list:
    :param K: K-fold cross-validation
    :return: 5 train-val pairs
    """
    splits = []
    patient_list.sort()
    kfold = KFold(n_splits=K, shuffle=shuffle, random_state=None)
    for i, (train_idx, test_idx) in enumerate(kfold.split(patient_list)):
        train_keys = np.array(patient_list)[train_idx]
        test_keys = np.array(patient_list)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    return splits


class AneuDataset(Dataset):
    def __init__(self, imgs_dir, aug=True):
        self.imgs_dir = imgs_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        self.aug = aug
        self.rupture_dir = '/home/xxx'
        self.rupture_file = load_rupture_ids(self.rupture_dir)
        self.ruptured = self.rupture_file[1]
        self.unruptured = self.rupture_file[0]

    def __len__(self):
        return len(self.ids)

    def is_rupture(self, name):
        if name in self.ruptured:
            return True
        elif name in self.unruptured:
            return False
        else:
            return None

    def random_trans(self, img):
        
        trans_img = img

        # random traslate
        if random.random() > 0.5:
            nonzero_idx = np.argwhere(trans_img>0)
            [maxx,maxy,maxz] = trans_img.shape
            roi_minx = np.min(nonzero_idx[:,0])
            roi_maxx = np.max(nonzero_idx[:,0])
            roi_miny = np.min(nonzero_idx[:,1])
            roi_maxy = np.max(nonzero_idx[:,1])
            roi_minz = np.min(nonzero_idx[:,2])
            roi_maxz = np.max(nonzero_idx[:,2])

            range_x = min(roi_minx-1,maxx-roi_maxx-1)
            range_y = min(roi_miny-1,maxy-roi_maxy-1)
            range_z = min(roi_minz-1,maxz-roi_maxz-1)

            if range_x > 0:
                roll_x = random.randint(1,range_x)
                trans_img = np.roll(trans_img,roll_x,axis=0)
            if range_y > 0:
                roll_y = random.randint(1,range_y)
                trans_img = np.roll(trans_img,roll_y,axis=1)
            if range_z > 0:
                roll_z = random.randint(1,range_z)
                trans_img = np.roll(trans_img,roll_z,axis=2)
            # print(roll_x,roll_y,roll_z)

        # random flip
        if random.random()>0.5:
            trans_img = trans_img[:,:,::-1]
        if random.random()>0.5:
            trans_img = trans_img[::-1,:,:]
        if random.random()>0.5:
            trans_img = trans_img[:,::-1,:]

        # random rotate 90
        if random.random() > 0.5:
            plane = random.random()
            if plane >0.66:
                trans_img = np.rot90(trans_img,axes=(0,1))
            elif plane > 0.33:
                trans_img = np.rot90(trans_img,axes=(0,2))
            else:
                trans_img = np.rot90(trans_img,axes=(1,2))

        return trans_img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = self.imgs_dir + '/' + idx + '.npy'
        # print(img_file)
        if not os.path.exists(img_file):
            img_file = self.imgs_dir + '/' + idx + '.nii'
        if '.npy' in img_file:
            img = np.load(img_file)
        elif '.nii' in img_file:
            img, _, _ = load_nii(img_file)

        if img.shape != (64,64,64):
            print('Error:',img_file)
        if np.sum(img) == 0:
            print('Mask Error:',img_file)
        if self.aug == True:
            trans_img = self.random_trans(img)
        else:
            trans_img = img
        
        img = np.expand_dims(img,axis=0)
        trans_img = np.expand_dims(trans_img,axis=0)

        img = torch.from_numpy(img.copy())
        trans_img = torch.from_numpy(trans_img.copy())
        return img, trans_img, idx

    def plot_3d(self,img1,img2):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = np.empty(img1.shape,dtype=object)
        link2 = (img2>0)
        colors[link2] = 'green'
        ax.voxels(img2,facecolors=colors,edgecolor='k')
        plt.show()


class RupturedDataset(Dataset):
    def __init__(self, imgs_dir=None, aug=True):
        self.imgs_dir = '/home/xxx'
        self.rupture_file = load_rupture_ids(self.imgs_dir, prefix=True)
        self.ruptured = self.rupture_file[1]
        self.ruptured.sort()
        self.unruptured = self.rupture_file[0]
        self.unruptured.sort()
        self.ids = self.ruptured + self.unruptured
        self.aug = aug

    def __len__(self):
        return len(self.ids)

    def random_trans(self, img):

        trans_img = img

        # random traslate
        if random.random() > 0.5:
            nonzero_idx = np.argwhere(trans_img > 0)
            [maxx, maxy, maxz] = trans_img.shape
            roi_minx = np.min(nonzero_idx[:, 0])
            roi_maxx = np.max(nonzero_idx[:, 0])
            roi_miny = np.min(nonzero_idx[:, 1])
            roi_maxy = np.max(nonzero_idx[:, 1])
            roi_minz = np.min(nonzero_idx[:, 2])
            roi_maxz = np.max(nonzero_idx[:, 2])

            range_x = min(roi_minx - 1, maxx - roi_maxx - 1)
            range_y = min(roi_miny - 1, maxy - roi_maxy - 1)
            range_z = min(roi_minz - 1, maxz - roi_maxz - 1)

            if range_x > 0:
                roll_x = random.randint(1, range_x)
                trans_img = np.roll(trans_img, roll_x, axis=0)
            if range_y > 0:
                roll_y = random.randint(1, range_y)
                trans_img = np.roll(trans_img, roll_y, axis=1)
            if range_z > 0:
                roll_z = random.randint(1, range_z)
                trans_img = np.roll(trans_img, roll_z, axis=2)

        # random flip
        if random.random() > 0.5:
            trans_img = trans_img[:, :, ::-1]
        if random.random() > 0.5:
            trans_img = trans_img[::-1, :, :]
        if random.random() > 0.5:
            trans_img = trans_img[:, ::-1, :]

        # random rotate 90
        if random.random() > 0.5:
            plane = random.random()
            if plane > 0.66:
                trans_img = np.rot90(trans_img, axes=(0, 1))
            elif plane > 0.33:
                trans_img = np.rot90(trans_img, axes=(0, 2))
            else:
                trans_img = np.rot90(trans_img, axes=(1, 2))

        return trans_img

    def __getitem__(self, i):
        p = random.random()
        label = 1
        if p < 0.5:
            idx = random.choice(self.ruptured)
        else:
            label = 0
            idx = random.choice(self.unruptured)
        img_file = self.imgs_dir + '/' + idx + '.npy'
        if not os.path.exists(img_file):
            img_file = self.imgs_dir + '/' + idx + '.nii'
        if '.npy' in img_file:
            img = np.load(img_file)
        elif '.nii' in img_file:
            img, _, _ = load_nii(img_file)

        if img.shape != (64, 64, 64):
            print('Error:', img_file)
        if np.sum(img) == 0:
            print('Mask Error:', img_file)
        if self.aug == True:
            trans_img = self.random_trans(img)
        else:
            trans_img = img

        img = np.expand_dims(img, axis=0)
        trans_img = np.expand_dims(trans_img, axis=0)

        img = torch.from_numpy(img.copy())
        trans_img = torch.from_numpy(trans_img.copy())
        return img, trans_img, label, idx

    def plot_3d(self, img1, img2):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = np.empty(img1.shape, dtype=object)
        link2 = (img2 > 0)
        colors[link2] = 'green'
        ax.voxels(img2, facecolors=colors, edgecolor='k')
        plt.show()


class RupturedDataset_ver2(Dataset):
    def __init__(self, s_img_ids=None, r_img_ids=None, attach_clinic=False, aug=True, train=True):
        self.imgs_dir = '/home/xxx'
        self.s_ids = s_img_ids
        self.r_ids = r_img_ids
        self.ids = s_img_ids + r_img_ids
        self.aug = aug
        self.train = train
        self.attach_clinic = attach_clinic
        if attach_clinic:
            self.clinic_data = read_csv(None)

    def __len__(self):
        return len(self.s_ids + self.r_ids)

    def random_trans(self, img):

        trans_img = img

        # random traslate
        if random.random() > 0.5:
            nonzero_idx = np.argwhere(trans_img > 0)
            [maxx, maxy, maxz] = trans_img.shape
            roi_minx = np.min(nonzero_idx[:, 0])
            roi_maxx = np.max(nonzero_idx[:, 0])
            roi_miny = np.min(nonzero_idx[:, 1])
            roi_maxy = np.max(nonzero_idx[:, 1])
            roi_minz = np.min(nonzero_idx[:, 2])
            roi_maxz = np.max(nonzero_idx[:, 2])

            range_x = min(roi_minx - 1, maxx - roi_maxx - 1)
            range_y = min(roi_miny - 1, maxy - roi_maxy - 1)
            range_z = min(roi_minz - 1, maxz - roi_maxz - 1)

            if range_x > 0:
                roll_x = random.randint(1, range_x)
                trans_img = np.roll(trans_img, roll_x, axis=0)
            if range_y > 0:
                roll_y = random.randint(1, range_y)
                trans_img = np.roll(trans_img, roll_y, axis=1)
            if range_z > 0:
                roll_z = random.randint(1, range_z)
                trans_img = np.roll(trans_img, roll_z, axis=2)

        # random flip
        if random.random() > 0.5:
            trans_img = trans_img[:, :, ::-1]
        if random.random() > 0.5:
            trans_img = trans_img[::-1, :, :]
        if random.random() > 0.5:
            trans_img = trans_img[:, ::-1, :]

        # random rotate 90
        if random.random() > 0.5:
            plane = random.random()
            if plane > 0.66:
                trans_img = np.rot90(trans_img, axes=(0, 1))
            elif plane > 0.33:
                trans_img = np.rot90(trans_img, axes=(0, 2))
            else:
                trans_img = np.rot90(trans_img, axes=(1, 2))

        return trans_img

    def __getitem__(self, i):
        if self.train:
            # randomly sample from two classes
            p = random.random()
            label = 1
            if p < 0.5:
                idx = random.choice(self.r_ids)
            else:
                label = 0
                idx = random.choice(self.s_ids)
        else:
            idx = self.ids[i]
            label = 1 if idx[0] == 'r' else 0
        img_file = self.imgs_dir + '/' + idx + '.npy'
        if not os.path.exists(img_file):
            img_file = self.imgs_dir + '/' + idx + '.nii'
        if '.npy' in img_file:
            img = np.load(img_file)
        elif '.nii' in img_file:
            img, _, _ = load_nii(img_file)

        if img.shape != (64, 64, 64):
            print('Error:', img_file)
        if np.sum(img) == 0:
            print('Mask Error:', img_file)
        if self.aug == True:
            trans_img = self.random_trans(img)
        else:
            trans_img = img

        img = np.expand_dims(img, axis=0)
        trans_img = np.expand_dims(trans_img, axis=0)
        label = np.expand_dims(label, axis=0)

        img = torch.from_numpy(img.copy())
        trans_img = torch.from_numpy(trans_img.copy())
        if self.attach_clinic:
            type = self.clinic_data[idx]['type']
            loc = self.clinic_data[idx]['loc']
            clinic_feature = np.concatenate((type, loc), axis=-1)
            return img, trans_img, label, idx, clinic_feature
        return img, trans_img, label, idx

    def plot_3d(self, img1, img2):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = np.empty(img1.shape, dtype=object)
        link2 = (img2 > 0)
        colors[link2] = 'green'
        ax.voxels(img2, facecolors=colors, edgecolor='k')
        plt.show()


class RupturedDataset_ver3(Dataset):
    def __init__(self, img_ids=None, aug=True, train=True):
        self.imgs_dir = '/home/xxx'
        self.ids = img_ids
        self.aug = aug
        self.train = train

    def __len__(self):
        return len(self.ids)

    def random_trans(self, img):

        trans_img = img

        # random traslate
        if random.random() > 0.5:
            nonzero_idx = np.argwhere(trans_img > 0)
            [maxx, maxy, maxz] = trans_img.shape
            roi_minx = np.min(nonzero_idx[:, 0])
            roi_maxx = np.max(nonzero_idx[:, 0])
            roi_miny = np.min(nonzero_idx[:, 1])
            roi_maxy = np.max(nonzero_idx[:, 1])
            roi_minz = np.min(nonzero_idx[:, 2])
            roi_maxz = np.max(nonzero_idx[:, 2])

            range_x = min(roi_minx - 1, maxx - roi_maxx - 1)
            range_y = min(roi_miny - 1, maxy - roi_maxy - 1)
            range_z = min(roi_minz - 1, maxz - roi_maxz - 1)

            if range_x > 0:
                roll_x = random.randint(1, range_x)
                trans_img = np.roll(trans_img, roll_x, axis=0)
            if range_y > 0:
                roll_y = random.randint(1, range_y)
                trans_img = np.roll(trans_img, roll_y, axis=1)
            if range_z > 0:
                roll_z = random.randint(1, range_z)
                trans_img = np.roll(trans_img, roll_z, axis=2)

        # random flip
        if random.random() > 0.5:
            trans_img = trans_img[:, :, ::-1]
        if random.random() > 0.5:
            trans_img = trans_img[::-1, :, :]
        if random.random() > 0.5:
            trans_img = trans_img[:, ::-1, :]

        # random rotate 90
        if random.random() > 0.5:
            plane = random.random()
            if plane > 0.66:
                trans_img = np.rot90(trans_img, axes=(0, 1))
            elif plane > 0.33:
                trans_img = np.rot90(trans_img, axes=(0, 2))
            else:
                trans_img = np.rot90(trans_img, axes=(1, 2))

        return trans_img

    def __getitem__(self, i):
        idx = self.ids[i]
        label = 1 if idx[0] == 'r' else 0

        img_file = self.imgs_dir + '/' + idx + '.npy'
        # print(img_file)
        if not os.path.exists(img_file):
            img_file = self.imgs_dir + '/' + idx + '.nii'
        if '.npy' in img_file:
            img = np.load(img_file)
        elif '.nii' in img_file:
            img, _, _ = load_nii(img_file)

        if img.shape != (64, 64, 64):
            print('Error:', img_file)
        if np.sum(img) == 0:
            print('Mask Error:', img_file)
        if self.aug == True:
            trans_img = self.random_trans(img)
        else:
            trans_img = img

        img = np.expand_dims(img, axis=0)
        trans_img = np.expand_dims(trans_img, axis=0)
        label = np.expand_dims(label, axis=0)

        img = torch.from_numpy(img.copy())
        trans_img = torch.from_numpy(trans_img.copy())
        return img, trans_img, label, idx

    def plot_3d(self, img1, img2):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = np.empty(img1.shape, dtype=object)
        link2 = (img2 > 0)
        colors[link2] = 'green'
        ax.voxels(img2, facecolors=colors, edgecolor='k')
        plt.show()


def random_trans(img):

    trans_img = img

    # random traslate
    if random.random() > 0.5:
        nonzero_idx = np.argwhere(trans_img > 0)
        [maxx, maxy, maxz] = trans_img.shape
        roi_minx = np.min(nonzero_idx[:, 0])
        roi_maxx = np.max(nonzero_idx[:, 0])
        roi_miny = np.min(nonzero_idx[:, 1])
        roi_maxy = np.max(nonzero_idx[:, 1])
        roi_minz = np.min(nonzero_idx[:, 2])
        roi_maxz = np.max(nonzero_idx[:, 2])

        range_x = min(roi_minx - 1, maxx - roi_maxx - 1)
        range_y = min(roi_miny - 1, maxy - roi_maxy - 1)
        range_z = min(roi_minz - 1, maxz - roi_maxz - 1)

        if range_x > 0:
            roll_x = random.randint(1, range_x)
            trans_img = np.roll(trans_img, roll_x, axis=0)
        if range_y > 0:
            roll_y = random.randint(1, range_y)
            trans_img = np.roll(trans_img, roll_y, axis=1)
        if range_z > 0:
            roll_z = random.randint(1, range_z)
            trans_img = np.roll(trans_img, roll_z, axis=2)
        # print(roll_x,roll_y,roll_z)

    # random flip
    if random.random() > 0.5:
        trans_img = trans_img[:, :, ::-1]
    if random.random() > 0.5:
        trans_img = trans_img[::-1, :, :]
    if random.random() > 0.5:
        trans_img = trans_img[:, ::-1, :]

    # random rotate 90
    if random.random() > 0.5:
        plane = random.random()
        if plane > 0.66:
            trans_img = np.rot90(trans_img, axes=(0, 1))
        elif plane > 0.33:
            trans_img = np.rot90(trans_img, axes=(0, 2))
        else:
            trans_img = np.rot90(trans_img, axes=(1, 2))

    return trans_img


def random_gen_data(imgs_dir, img_ids, batch_size=2):
    choose_batch = np.random.choice(img_ids, batch_size)
    trans_img_list = []
    img_list = []
    r = True
    for idx in choose_batch:
        if idx[0] == 's':
            r = False
        img_file = imgs_dir + '/' + idx + '.npy'
        if not os.path.exists(img_file):
            img_file = imgs_dir + '/' + idx + '.nii'
        if '.npy' in img_file:
            img = np.load(img_file)
        elif '.nii' in img_file:
            img, _, _ = load_nii(img_file)
        trans_img_list.append(random_trans(img))
        img_list.append(img)
    batch_trans_img = np.stack(trans_img_list, axis=0)
    batch_img = np.stack(img_list, axis=0)
    label = np.ones(batch_trans_img.shape[0], dtype=np.uint8) if r is True \
        else np.zeros(batch_trans_img.shape[0], dtype=np.uint8)
    return np.expand_dims(batch_img, axis=1), np.expand_dims(batch_trans_img, axis=1), \
           np.expand_dims(label, axis=1), choose_batch


def read_csv(file_path=None):
    S_arr = np.array([0], dtype=np.float)
    B_arr = np.array([1], dtype=np.float)
    MCA_arr = np.array([1, 0, 0, 0, 0, 0], dtype=np.float)
    AComA_arr = np.array([0, 1, 0, 0, 0, 0], dtype=np.float)
    PComA_arr = np.array([0, 0, 1, 0, 0, 0], dtype=np.float)
    BA_arr = np.array([0, 0, 0, 1, 0, 0], dtype=np.float)
    ICA_arr = np.array([0, 0, 0, 0, 1, 0], dtype=np.float)
    PCA_arr = np.array([0, 0, 0, 0, 0, 1], dtype=np.float)

    TYPE = {'S': S_arr, 'B': B_arr}
    LOC = {'MCA': MCA_arr, 'AComA': AComA_arr, 'PComA': PComA_arr, 'BA': BA_arr, 'ICA': ICA_arr, 'PCA': PCA_arr}

    data = {}
    if file_path is None:
        file_path = '../deepfeature_gan.csv'
    csv_reader = csv.reader(open(file_path))
    i = 0
    for line in csv_reader:
        if i == 0:
            i += 1
            continue
        # print(line[-9], line[-2], line[-1])
        data[line[-9]] = {}
        data[line[-9]]['type'] = TYPE[line[-2]]
        data[line[-9]]['loc'] = LOC[line[-1]]
    #
    data['s_WO78US_mask1'] = {}
    data['s_WO78US_mask1']['type'] = TYPE['B']
    data['s_WO78US_mask1']['loc'] = LOC['AComA']
    data['s_WO69US_mask1'] = {}
    data['s_WO69US_mask1']['type'] = TYPE['S']
    data['s_WO69US_mask1']['loc'] = LOC['PComA']
    data['s_WO78US_mask2'] = {}
    data['s_WO78US_mask2']['type'] = TYPE['S']
    data['s_WO78US_mask2']['loc'] = LOC['ICA']
    return data

