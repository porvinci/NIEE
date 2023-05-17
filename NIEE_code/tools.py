import numpy as np
import scipy
import torch
import pathlib
import dgl
import os
import pandas as pd
import pickle as pkl
import random
import tqdm
import time
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, average_precision_score, ndcg_score

# 全部的数据用于训练，dis:144用作case study
def split_case_study(prefix, hg, etype_name, drug_di_src, drug_di_dst, drug_di_link):
    pos_label=[1]*drug_di_link
    pos_data=list(zip(drug_di_src,drug_di_dst,pos_label))
    drug_di_adj = np.array(hg.adj(etype=etype_name).to_dense())
    full_idx = np.where(drug_di_adj==0)
    sample = random.sample(range(0, len(full_idx[0])), drug_di_link)
    neg_label = [0]*drug_di_link
    neg_data = list(zip(full_idx[0][sample],full_idx[1][sample],neg_label))
    full_data = pos_data + neg_data
    random.shuffle(full_data)

    train_data = np.array(full_data)

    # Alzheimer: alz
    alz_pos_idx = np.where(drug_di_adj.T[144]==1)
    alz_pos_label=[1]*alz_pos_idx[0].shape[0]
    alz_idx = [144]*alz_pos_idx[0].shape[0]
    alz_pos_data = list(zip(alz_pos_idx[0], alz_idx, alz_pos_label))

    alz_neg_idx = np.where(drug_di_adj.T[144]==0)
    alz_neg_label=[0]*alz_neg_idx[0].shape[0]
    alz_idx = [144]*alz_neg_idx[0].shape[0]
    alz_neg_data = list(zip(alz_neg_idx[0], alz_idx, alz_neg_label))
    
    test_data = np.array(alz_pos_data + alz_neg_data)
    eval_size = int(len(full_data)*0.05)
    eval_data = np.array(train_data[:eval_size])

    with open(os.path.join(prefix, 'mydata_all_train.pkl'), 'wb') as file:
        pkl.dump(train_data, file)
    with open(os.path.join(prefix, 'mydata_eval.pkl'), 'wb') as file:
        pkl.dump(eval_data, file)
    with open(os.path.join(prefix, 'alz_test.pkl'), 'wb') as file:
        pkl.dump(test_data, file)
    return train_data, eval_data, test_data



def split_data(prefix, hg, etype_name, drug_di_src, drug_di_dst, drug_di_link):
    pos_label=[1]*drug_di_link
    pos_data=list(zip(drug_di_src,drug_di_dst,pos_label))
    drug_di_adj = np.array(hg.adj(etype=etype_name).to_dense())
    full_idx = np.where(drug_di_adj==0)
    sample = random.sample(range(0, len(full_idx[0])), drug_di_link)
    neg_label = [0]*drug_di_link
    neg_data = list(zip(full_idx[0][sample],full_idx[1][sample],neg_label))
    full_data = pos_data + neg_data
    random.shuffle(full_data)

    train_size = int(len(full_data) * 0.7)
    eval_size = int(len(full_data) * 0.1)
    test_size = len(full_data) - train_size - eval_size
    train_data = full_data[:train_size]
    eval_data = full_data[train_size : train_size+eval_size]
    test_data = full_data[train_size+eval_size : train_size+eval_size+test_size]
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    test_data = np.array(test_data)
    with open(os.path.join(prefix, 'mydata_train.pkl'), 'wb') as file:
        pkl.dump(train_data, file)
    with open(os.path.join(prefix, 'mydata_eval.pkl'), 'wb') as file:
        pkl.dump(eval_data, file)
    with open(os.path.join(prefix, 'mydata_test.pkl'), 'wb') as file:
        pkl.dump(test_data, file)
    
    return train_data, eval_data, test_data


def generate_metapath(hg,head,meta_paths,path_names,path,name,_num_walks_per_node, _walk_length):
    dict={}
    for meta_path, path_name in zip(meta_paths, path_names):
        dict[path_name]={}
        for idx in tqdm.trange(hg.number_of_nodes(head)):
            traces, _ = dgl.sampling.random_walk(
                hg, [idx] * _num_walks_per_node, metapath=meta_path * _walk_length)
            dict[path_name][idx]=traces.long()
    with open(os.path.join(path, name), 'wb') as file:
        pkl.dump(dict, file)


def evaluate_auc(pred, label):
    if np.sum(label) == 0:
        m=[1]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    if np.sum(label) == len(label):
        m=[0]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    res=roc_auc_score(y_score=pred, y_true=label)
    return res

def evaluate_ap(pred, label):
    if np.sum(label) == 0:
        m=[1]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    if np.sum(label) == len(label):
        m=[0]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    res=average_precision_score(y_score=pred, y_true=label)
    return res

def evaluate_acc(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return accuracy_score(y_pred=res, y_true=label)

def evaluate_f1_score(pred, label):
    res = []
    for _value in pred:
        if _value >= 0.5:
            res.append(1)
        else:
            res.append(0)
    return f1_score(y_pred=res, y_true=label)

def evaluate_logloss(pred, label):
    if np.sum(label) == 0:
        m=[1]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])
    if np.sum(label) == len(label):
        m=[0]
        label=np.vstack([label,m])
        pred=np.vstack([pred,m])    
    res = log_loss(y_true=label, y_pred=pred,eps=1e-7, normalize=True)
    return res