from networks import Encoder3D,Classifier3D,R3D_18_Encoder,R3D_18_Encoder_Split, R3D_18_Encoder_Split_ver2
from loader_ver2 import AneuDataset, RupturedDataset_ver2, RupturedDataset_ver3
from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from loss import DiceLoss
import numpy as np
from torch.autograd import Function
from test import model_test
from pytorch_metric_learning.losses import NTXentLoss, NPairsLoss
from sklearn.model_selection import KFold
from collections import OrderedDict
from os import listdir
import os
import torch.backends.cudnn as cudnn
import time
from utils import print_log
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score,accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# only BCE

np.random.seed(12345)
torch.manual_seed(12345)
if torch.cuda.is_available():
    print('torch.cuda.is_available()')
    torch.cuda.manual_seed_all(12345)
cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    # sort patient_list to ensure the splited data unchangeable every time.
    patient_list.sort()
    # k-fold, I think it doesn't matter whether the shuffle is true or not
    kfold = KFold(n_splits=K, shuffle=shuffle, random_state=None)
    for i, (train_idx, test_idx) in enumerate(kfold.split(patient_list)):
        train_keys = np.array(patient_list)[train_idx]
        test_keys = np.array(patient_list)[test_idx]
        splits.append(OrderedDict())
        splits[-1]['train'] = train_keys
        splits[-1]['val'] = test_keys
    return splits


def cal_dice(input,target,threshold=0.5):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    input = (input > threshold).float()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return s / (i + 1)


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        # self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t


def train(cv=0, batch_size=4, log_path_=None, save_checkpoint=None):
    # training protocol
    batch_size = batch_size
    lr = 0.001
    con_temperature = 0.05
    weight_decay = 1e-4
    lr_step_size = 100
    lr_gamma = 0.1
    TOTAL_EPOCHS = 200
    LOG_PATH = log_path_
    CHECKPOINT_PATH = save_checkpoint
    SUMMARY_WRITER = log_path_

    # initialize log
    now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    log_path = LOG_PATH
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    log = open(os.path.join(log_path, 'log_{}_{}_{}.txt'.format(now, cv, batch_size)), 'w')
    print_log('save path : {}'.format(log_path), log)

    # initialize summarywriter
    writer = SummaryWriter(SUMMARY_WRITER)

    # prepare data
    rupture_dir = '/home/xxx'
    cta_dir = '/home/xxx'

    rupture_file = load_rupture_ids(rupture_dir, prefix=True)
    ruptured = rupture_file[1]
    ruptured.sort()
    unruptured = rupture_file[0]
    unruptured.sort()

    ruptured_splits = split_data(ruptured, K=5, shuffle=False)
    unruptured_splits = split_data(unruptured, K=5, shuffle=False)

    order = cv
    train_rupture_dataset = RupturedDataset_ver2(s_img_ids=list(unruptured_splits[order]['train']),
                                                 r_img_ids=list(ruptured_splits[order]['train']),
                                                 aug=True, train=True)
    val_rupture_dataset = RupturedDataset_ver2(s_img_ids=list(unruptured_splits[order]['val']),
                                               r_img_ids=list(ruptured_splits[order]['val']),
                                               aug=False, train=False)

    train_loader = DataLoader(train_rupture_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_rupture_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    net = R3D_18_Encoder_Split_ver2()
    net.cuda()

    classifier_criterion = torch.nn.BCEWithLogitsLoss()
    con_criterion = NTXentLoss(con_temperature)

    optimizer1 = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999])
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, lr_step_size, lr_gamma)

    best_auc = 0.
    best_acc = 0.
    for epoch in range(TOTAL_EPOCHS):
        it = -1
        net.train()
        for img, trans_img, label, idx in train_loader:
            loss1_value = 0.
            loss2_value = 0.
            it += 1
            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
                trans_img = trans_img.float().cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            optimizer1.zero_grad()

            logits, embedding_feature = net(trans_img)

            label = label.type_as(logits)
            loss1 = classifier_criterion(logits, label)
            loss1_value += loss1.data.cpu().numpy()

            loss = loss1
            loss.backward()
            optimizer1.step()

            log_str = 'epoch: {}, iter: {}, lr: {:.8f}, loss1: {:.6f}, loss2: {:.6f}' \
                .format(epoch, it, scheduler1.get_lr()[0], loss1_value, loss2_value)
            print_log(log_str, log)

        scheduler1.step()

        # validation
        with torch.no_grad():
            net.eval()
            preds = None
            labels = None
            probs = None
            for img, trans_img, label, idx in val_loader:
                if torch.cuda.is_available():
                    img = img.float().cuda(non_blocking=True)
                    trans_img = trans_img.float().cuda(non_blocking=True)
                logits, embedding_feature = net(img)
                prob = torch.sigmoid(logits).cpu().numpy()
                pred = np.round_(prob)
                if preds is None:
                    preds = pred
                else:
                    preds = np.concatenate((preds, pred), axis=0)
                if labels is None:
                    labels = label
                else:
                    labels = np.concatenate((labels, label), axis=0)
                if probs is None:
                    probs = prob
                else:
                    probs = np.concatenate((probs, prob), axis=0)
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, probs)
            ap = average_precision_score(labels, probs)
            log_str = 'cv: {}, epoch: {}, acc: {}, auc: {}, ap: {}'.format(cv, epoch, acc, auc, ap)
            print_log(log_str, log)
            if auc > best_auc:
                best_auc = auc
                torch.save({'model': net.state_dict()}, CHECKPOINT_PATH + '/checkpoint_cv{}_epoch{}_bs{}.pth'.format(cv, epoch, batch_size))
                print_log('save best!', log)
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': net.state_dict()}, CHECKPOINT_PATH + '/checkpoint_cv{}_epoch{}_bs{}.pth'.format(cv, epoch, batch_size))
                print_log('save best!', log)


def val(cv=0, batch_size=4, log_path_=None, save_checkpoint=None, epoch=-1):
    # training protocol
    batch_size = batch_size
    lr = 0.001
    con_temperature = 0.05
    weight_decay = 1e-4
    lr_step_size = 100
    lr_gamma = 0.1
    TOTAL_EPOCHS = 200
    LOG_PATH = log_path_
    CHECKPOINT_PATH = save_checkpoint + '/checkpoint_cv{}_epoch{}_bs{}.pth'.format(cv, epoch, batch_size)
    SUMMARY_WRITER = log_path_

    # prepare data
    rupture_dir = 'xxx'
    cta_dir = 'xxx'

    rupture_file = load_rupture_ids(rupture_dir, prefix=True)
    ruptured = rupture_file[1]
    ruptured.sort()
    unruptured = rupture_file[0]
    unruptured.sort()

    ruptured_splits = split_data(ruptured, K=5, shuffle=False)
    unruptured_splits = split_data(unruptured, K=5, shuffle=False)

    order = cv
    
    train_rupture_dataset = RupturedDataset_ver2(s_img_ids=list(unruptured_splits[order]['train']),
                                                 r_img_ids=list(ruptured_splits[order]['train']),
                                                 aug=True, train=True)
    val_rupture_dataset = RupturedDataset_ver2(s_img_ids=list(unruptured_splits[order]['val']),
                                               r_img_ids=list(ruptured_splits[order]['val']),
                                               aug=False, train=False)

    train_loader = DataLoader(train_rupture_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_rupture_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    net = R3D_18_Encoder_Split_ver2()
    net.cuda()

    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        net.eval()
        preds = None
        labels = None
        probs = None
        for img, trans_img, label, idx in val_loader:
            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
                trans_img = trans_img.float().cuda(non_blocking=True)
            logits, embedding_feature = net(img)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = np.round_(prob)
            if preds is None:
                preds = pred
            else:
                preds = np.concatenate((preds, pred), axis=0)
            if labels is None:
                labels = label
            else:
                labels = np.concatenate((labels, label), axis=0)
            if probs is None:
                probs = prob
            else:
                probs = np.concatenate((probs, prob), axis=0)
        
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, probs)
        log_str = 'cv: {}, epoch: {}, acc: {}, auc: {}, ap: {}'.format(cv, epoch, acc, auc, ap)
        print(log_str)


if __name__ == '__main__':
    train(0, batch_size=4, log_path_='./logs', save_checkpoint='./checkpoints')
    # train(1, batch_size=4, log_path_='./logs', save_checkpoint='./checkpoints')
    # train(2, batch_size=4, log_path_='./logs', save_checkpoint='./checkpoints')
    # train(3, batch_size=4, log_path_='./logs', save_checkpoint='./checkpoints')
    # train(4, batch_size=4, log_path_='./logs', save_checkpoint='./checkpoints')
