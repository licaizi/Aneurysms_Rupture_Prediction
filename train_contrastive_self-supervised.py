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
import torch.nn.functional as F

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
    # print(len(filenames), filenames)   # 121
    rupture_file = {}
    rupture_file[1] = []
    rupture_file[0] = []
    for filename in filenames:
        tmp_name = filename
        is_rupture = 1 if tmp_name[0] == 'r' else 0
        # stop = tmp_name.index('.nii')
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
    dataset = AneuDataset('/home/xxx')

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    net = R3D_18_Encoder_Split_ver2()
    net.cuda()

    con_criterion = NTXentLoss(con_temperature)

    optimizer1 = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999])
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, lr_step_size, lr_gamma)

    best_acc = 0.
    for epoch in range(TOTAL_EPOCHS):
        it = -1
        net.train()
        for img, trans_img, idx in train_loader:
            loss1_value = 0.
            loss2_value = 0.
            it += 1

            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
                trans_img = trans_img.float().cuda(non_blocking=True)

            optimizer1.zero_grad()

            logits, embedding_feature = net(img)
            trans_logits, trans_embedding_feature = net(trans_img)

            label = torch.arange(embedding_feature.size(0))

            embeddings = torch.cat((F.normalize(embedding_feature, p=2, dim=1),
                                    F.normalize(trans_embedding_feature, p=2, dim=1)), dim=0)
            labels = torch.cat((label, label), dim=0).cuda()
            loss2 = con_criterion(embeddings, labels)
            loss2_value += loss2.data.cpu().numpy()

            loss = loss2
            loss.backward()
            optimizer1.step()

            log_str = 'epoch: {}, iter: {}, lr: {:.8f}, loss1: {:.6f}, loss2: {:.6f}' \
                .format(epoch, it, scheduler1.get_lr()[0], loss1_value, loss2_value)
            print_log(log_str, log)

        scheduler1.step()

        torch.save({'model': net.state_dict()}, CHECKPOINT_PATH + '/checkpoint_cv{}_epoch{}_bs{}.pth'.format(cv, epoch, batch_size))

        # do tsne
        net.eval()
        embedding_eval = torch.zeros((0, 64), dtype=torch.float)
        with torch.no_grad():
            for img, trans_img, idx in train_loader:
                if torch.cuda.is_available():
                    img = img.float().cuda(non_blocking=True)
                logits, embedding_feature = net(img)
                embedding_eval = torch.cat([embedding_eval, embedding_feature.cpu()])
                torch.cuda.empty_cache()
            writer.add_embedding(embedding_eval.data, global_step=epoch)

    writer.close()


def val(cv=0, batch_size=4, log_path_=None, save_checkpoint=None, epoch=0):
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

    val_rupture_dataset = RupturedDataset_ver2(s_img_ids=list(unruptured),
                                               r_img_ids=list(ruptured),
                                               aug=False, train=False)
    val_loader = DataLoader(val_rupture_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    net = R3D_18_Encoder_Split_ver2()
    net.cuda()
    checkpoint = torch.load(CHECKPOINT_PATH + '/checkpoint_cv{}_epoch{}_bs{}.pth'.format(cv, epoch, batch_size))
    net.load_state_dict(checkpoint['model'])

    net.eval()
    embedding_eval = torch.zeros((0, 64), dtype=torch.float)
    embedding_label = torch.zeros((0, 1), dtype=torch.uint8)
    img_labels = []
    with torch.no_grad():
        for img, trans_img, label, idx in val_loader:
            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
            logits, embedding_feature = net(img)
            embedding_eval = torch.cat([embedding_eval, embedding_feature.cpu()])
            embedding_label = torch.cat([embedding_label, label])
            for id in idx:
                if id[0] == 's':
                    img_labels.append(0)
                elif id[0] == 'r':
                    img_labels.append(1)
            torch.cuda.empty_cache()
        embedding_label = torch.squeeze(embedding_label)
        print(embedding_eval.shape)
        writer.add_embedding(embedding_eval.data, metadata=embedding_label.data, global_step=epoch)
        writer.close()
    return embedding_eval, img_labels


if __name__ == '__main__':
    train(1, batch_size=4, log_path_='logs_unsupervised', save_checkpoint='./checkpoints_unsupervised')
    

