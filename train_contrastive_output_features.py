from networks import R3D_18_Encoder_Split, R3D_18_Encoder_Split_ver2, R3D_18_Encoder_Split_ver2_finetune
from loader_ver2 import AneuDataset, RupturedDataset_ver2, RupturedDataset_ver3
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.model_selection import KFold
from collections import OrderedDict
from os import listdir
import os
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score,accuracy_score
from checkpoints import CKPTS
import csv


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


def val(cv=0, batch_size=4, log_path_=None, save_checkpoint=None, model_path=None, saved_ckpt=None, save_path=None):
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
                                               aug=False, train=False, attach_clinic=True)

    val_loader = DataLoader(val_rupture_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    net = R3D_18_Encoder_Split_ver2_finetune(model_ckpt=model_path, linear=2, frozen=False, attach_clinic=True, attach_contrastive=True)
    net.cuda()
    if saved_ckpt is not None:
        ckpt = torch.load(saved_ckpt)
        net.load_state_dict(ckpt['model'])

    with torch.no_grad():
        net.eval()
        preds = None
        labels = None
        probs = None
        embeddings_vector = None
        last_layer_embeddings_vector = None

        namelist = []
        fieldlist = []
        fieldlist.append('name')
        fieldlist.append('label')
        fieldlist.append('probs')
        for i in range(64):
            fieldlist.append('f'+str(i+1))
        for i in range(64):
            fieldlist.append('lf'+str(i+1))

        for img, trans_img, label, idx, clinic_feature in val_loader:
            if torch.cuda.is_available():
                img = img.float().cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                label = label.cpu().numpy()
                clinic_feature = clinic_feature.float().cuda(non_blocking=True)
            logits, embeddings, last_layer_feature = net(img, clinic_feature=clinic_feature)
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
            embeddings = embeddings.cpu().numpy()
            if embeddings_vector is None:
                embeddings_vector = embeddings
            else:
                embeddings_vector = np.concatenate((embeddings_vector, embeddings), axis=0)

            last_layer_feature = last_layer_feature.cpu().numpy()
            if last_layer_embeddings_vector is None:
                last_layer_embeddings_vector = last_layer_feature
            else:
                last_layer_embeddings_vector = np.concatenate((last_layer_embeddings_vector, last_layer_feature), axis=0)

            for i in range(len(idx)):
                namelist.append(idx[i])
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
        log_str = 'cv: {}, epoch: {}, acc: {}, auc: {}, ap: {}'.format(cv, 0, acc, auc, ap)
        print(log_str)
        with open(save_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fieldlist)
            for line in range(len(val_rupture_dataset)):
                content = []
                content.append(namelist[line])
                content.append(labels[line][0])
                content.append(probs[line][0])
                for i in range(len(embeddings_vector[line])):
                    content.append(embeddings_vector[line][i])
                for i in range(len(last_layer_embeddings_vector[line])):
                    content.append(last_layer_embeddings_vector[line][i])
                csvwriter.writerow(content)


if __name__ == '__main__':
    # output feature vectors and saved as 'csv'
    model_path = CKPTS['self_sup']
    save_path_cv1 = './saved_ckpt/feature_cv1.csv'
    save_path_cv2 = './saved_ckpt/feature_cv2.csv'
    save_path_cv3 = './saved_ckpt/feature_cv3.csv'
    save_path_cv4 = './saved_ckpt/feature_cv4.csv'
    save_path_cv5 = './saved_ckpt/feature_cv5.csv'
    saved_pth_cv1_clinic = CKPTS['finetune_clinic_cv1']
    saved_pth_cv2_clinic = CKPTS['finetune_clinic_cv2']
    saved_pth_cv3_clinic = CKPTS['finetune_clinic_cv3']
    saved_pth_cv4_clinic = CKPTS['finetune_clinic_cv4']
    saved_pth_cv5_clinic = CKPTS['finetune_clinic_cv5']

    val(0, batch_size=4, log_path_='./saved_ckpt',
        save_checkpoint='./saved_ckpt', model_path=model_path, saved_ckpt=saved_pth_cv1_clinic, save_path=save_path_cv1)
    val(1, batch_size=4, log_path_='./saved_ckpt',
        save_checkpoint='./saved_ckpt', model_path=model_path, saved_ckpt=saved_pth_cv2_clinic, save_path=save_path_cv2)
    val(2, batch_size=4, log_path_='./saved_ckpt',
        save_checkpoint='./saved_ckpt', model_path=model_path, saved_ckpt=saved_pth_cv3_clinic, save_path=save_path_cv3)
    val(3, batch_size=4, log_path_='./saved_ckpt',
        save_checkpoint='./saved_ckpt', model_path=model_path, saved_ckpt=saved_pth_cv4_clinic, save_path=save_path_cv4)
    val(4, batch_size=4, log_path_='./saved_ckpt',
        save_checkpoint='./saved_ckpt', model_path=model_path, saved_ckpt=saved_pth_cv5_clinic, save_path=save_path_cv5)


