import pickle
import time
import numpy as np
import scipy
import torch
import pathlib
import dgl
import os
import pandas as pd
import pickle as pkl
import tools
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, drug_pth, dis_pth, drug_metas, dis_metas, x_data, y_data):
        self.drug_pth = drug_pth
        self.dis_pth = dis_pth
        self.RI = drug_pth[drug_metas[0]]
        self.IR = dis_pth[dis_metas[0]]
        self.RIRI = drug_pth[drug_metas[1]]
        self.IRIR = dis_pth[dis_metas[1]]
        self.RIGI = drug_pth[drug_metas[2]]
        self.IGIR = dis_pth[dis_metas[2]]
        self.x_data = x_data
        self.y_data = torch.Tensor(y_data).unsqueeze(1)
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        return self.RI[self.x_data[index][0]] + 1, self.IR[self.x_data[index][1]] + 1, \
               self.RIRI[self.x_data[index][0]] + 1, self.IRIR[self.x_data[index][1]] + 1, \
               self.RIGI[self.x_data[index][0]] + 1, self.IGIR[self.x_data[index][1]] + 1, \
               self.y_data[index]

    def __len__(self):
        return self.len


# 所有的bioDDG数据均用来训练模型
def load_bioDDG_ALL(batch_size, _num_walks_per_node=16, prefix='data/JYdata', _walk_length=1 ):
    # 1.建异构图
    _data_list = ['raw/drug_di.csv', 'raw/di_gene.csv']
    if not (os.path.exists(os.path.join(prefix, 'processed/mydata_hg.pkl'))):
        # disease_gene
        dg = pd.read_csv(os.path.join(prefix, _data_list[1]), header=None).values - 1
        disease_gene_src = dg[:, 0].tolist()
        disease_gene_dst = dg[:, 1].tolist()
        # drug_disease
        drug_di_link = 0
        drdi = pd.read_csv(os.path.join(prefix, _data_list[0]), header=None).values - 1
        drug_di_src = drdi[:, 0].tolist()
        drug_di_dst = drdi[:, 1].tolist()
        drug_di_link = drdi.shape[0]
        # build graph
        hg = dgl.heterograph({
            ('disease', 'ig', 'gene'): (disease_gene_src, disease_gene_dst),
            ('gene', 'gi', 'disease'): (disease_gene_dst, disease_gene_src),
            ('drug', 'ri', 'disease'): (drug_di_src, drug_di_dst),
            ('disease', 'ir', 'drug'): (drug_di_dst, drug_di_src)})
        # print(hg)
        with open(os.path.join(prefix, 'processed/mydata_hg.pkl'), 'wb') as file:
            pkl.dump(hg, file)
        print("Graph constructed.")
    else:
        hg_file = open(os.path.join(prefix, 'processed/mydata_hg.pkl'), 'rb')
        hg = pkl.load(hg_file)
        hg_file.close()
        print("Graph Loaded.")

    # 2.划分数据集。用所有的数据集去训练，无需验证集，测试集里面是case study的药物 | dis编号144 阿兹海默症
    etype_name = 'ri'  # predict edge type
    if not (os.path.exists(os.path.join(prefix, 'processed/mydata_all_train.pkl'))):
        train_data, eval_data, test_data = tools.split_case_study(prefix+'/processed', hg, etype_name, drug_di_src, drug_di_dst,
                                                           drug_di_link)
        print("Train, eval and test splited.")
    else:
        train_file = open(prefix + '/processed/mydata_all_train.pkl', 'rb')
        train_data = pkl.load(train_file)
        train_file.close()
        eval_file = open(prefix + '/processed/mydata_eval.pkl', 'rb')
        eval_data = pkl.load(eval_file)
        eval_file.close()
        test_file = open(prefix + '/processed/alz_test.pkl', 'rb')
        test_data = pkl.load(test_file)
        test_file.close()
        
        print("Train, eval and test loaded.")
    # 3.(用所有数据去)生成元路径序列
    # 定义元路径
    scale = '_' + str(_num_walks_per_node) # + '_' + str(_walk_length)
    drug_paths = [['ri'], ['ri', 'ir', 'ri'], ['ri', 'ig', 'gi']]
    disease_paths = [['ir'], ['ir', 'ri', 'ir'], ['ig', 'gi', 'ir']]
    drug_metas = ['RI', 'RIRI', 'RIGI']
    disease_metas = ['IR', 'IRIR', 'IGIR']
    drug_pkl = 'mydata_drug' + scale + '.pkl'
    disease_pkl = 'mydata_disease' + scale + '.pkl'

    if not (os.path.exists(os.path.join(prefix+'/processed', drug_pkl))):
        tools.generate_metapath(hg, 'drug', drug_paths, drug_metas, prefix+'/processed', drug_pkl, _num_walks_per_node, _walk_length)
        tools.generate_metapath(hg, 'disease', disease_paths, disease_metas, prefix+'/processed', disease_pkl, _num_walks_per_node, _walk_length)
        print("Paths sampled.")


    print("Load paths from:")
    print(drug_pkl)
    print(disease_pkl)
    drug_file = open(prefix+'/processed/' + drug_pkl, 'rb')
    drug_pth = pkl.load(drug_file)
    drug_file.close()
    disease_file = open(prefix+'/processed/' + disease_pkl, 'rb')
    disease_pth = pkl.load(disease_file)
    disease_file.close()
    train_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, train_data[:, :2], train_data[:, 2])
    eval_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, eval_data[:, :2], eval_data[:, 2])
    test_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, test_data[:, :2], test_data[:, 2])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader




def load_bioDDG(batch_size, _num_walks_per_node=16, prefix='data/bioDDG', _walk_length=1 ):
    # 1.建异构图
    _data_list = ['raw/drug_di.csv', 'raw/di_gene.csv']
    if not (os.path.exists(os.path.join(prefix, 'processed/mydata_hg.pkl'))):
        # disease_gene
        dg = pd.read_csv(os.path.join(prefix, _data_list[1]), header=None).values - 1
        disease_gene_src = dg[:, 0].tolist()
        disease_gene_dst = dg[:, 1].tolist()
        # drug_disease
        drug_di_link = 0
        drdi = pd.read_csv(os.path.join(prefix, _data_list[0]), header=None).values - 1
        drug_di_src = drdi[:, 0].tolist()
        drug_di_dst = drdi[:, 1].tolist()
        drug_di_link = drdi.shape[0]
        # build graph
        hg = dgl.heterograph({
            ('disease', 'ig', 'gene'): (disease_gene_src, disease_gene_dst),
            ('gene', 'gi', 'disease'): (disease_gene_dst, disease_gene_src),
            ('drug', 'ri', 'disease'): (drug_di_src, drug_di_dst),
            ('disease', 'ir', 'drug'): (drug_di_dst, drug_di_src)})
        # print(hg)
        with open(os.path.join(prefix, 'processed/mydata_hg.pkl'), 'wb') as file:
            pkl.dump(hg, file)
        print("Graph constructed.")
    else:
        hg_file = open(os.path.join(prefix, 'processed/mydata_hg.pkl'), 'rb')
        hg = pkl.load(hg_file)
        hg_file.close()
        print("Graph Loaded.")

    # 2.划分数据集
    etype_name = 'ri'  # predict edge type
    if not (os.path.exists(os.path.join(prefix, 'processed/mydata_train.pkl'))):
        train_data, eval_data, test_data = tools.split_data(prefix+'/processed', hg, etype_name, drug_di_src, drug_di_dst,
                                                           drug_di_link)
        print("Train, eval, and test splited.")
    else:
        train_file = open(prefix + '/processed/mydata_train.pkl', 'rb')
        train_data = pkl.load(train_file)
        train_file.close()
        eval_file = open(prefix + '/processed/mydata_eval.pkl', 'rb')
        eval_data = pkl.load(eval_file)
        eval_file.close()
        test_file = open(prefix + '/processed/mydata_test.pkl', 'rb')
        test_data = pkl.load(test_file)
        test_file.close()
        
        print("Train, eval, and test loaded.")
    # 3.(用训练集去)生成元路径序列
    # 定义元路径
    scale = '_' + str(_num_walks_per_node) # + '_' + str(_walk_length)
    drug_paths = [['ri'], ['ri', 'ir', 'ri'], ['ri', 'ig', 'gi']]
    disease_paths = [['ir'], ['ir', 'ri', 'ir'], ['ig', 'gi', 'ir']]
    drug_metas = ['RI', 'RIRI', 'RIGI']
    disease_metas = ['IR', 'IRIR', 'IGIR']
    drug_pkl = 'mydata_drug' + scale + '.pkl'
    disease_pkl = 'mydata_disease' + scale + '.pkl'
    if not (os.path.exists(os.path.join(prefix, 'processed/mydata_hg_train.pkl'))):
        idx = np.where(train_data[:,-1]==1)
        drug_di_src = train_data[idx, 0][0]
        drug_di_dst = train_data[idx, 1][0]
        # build graph
        hg_train = dgl.heterograph({
            ('disease', 'ig', 'gene'): (disease_gene_src, disease_gene_dst),
            ('gene', 'gi', 'disease'): (disease_gene_dst, disease_gene_src),
            ('drug', 'ri', 'disease'): (drug_di_src, drug_di_dst),
            ('disease', 'ir', 'drug'): (drug_di_dst, drug_di_src)})
        # print(hg_train)
        # time.sleep(1000)
        with open(os.path.join(prefix, 'processed/mydata_hg_train.pkl'), 'wb') as file:
            pkl.dump(hg, file)
        print("Train Graph constructed.")
    else:
        hg_file = open(os.path.join(prefix, 'processed/mydata_hg_train.pkl'), 'rb')
        hg_train = pkl.load(hg_file)
        hg_file.close()
        print("Train Graph Loaded.")

    if not (os.path.exists(os.path.join(prefix+'/processed', drug_pkl))):
        tools.generate_metapath(hg_train, 'drug', drug_paths, drug_metas, prefix+'/processed', drug_pkl, _num_walks_per_node, _walk_length)
        tools.generate_metapath(hg_train, 'disease', disease_paths, disease_metas, prefix+'/processed', disease_pkl, _num_walks_per_node, _walk_length)
        print("Paths sampled.")


    print("Load paths from:")
    print(drug_pkl)
    print(disease_pkl)
    drug_file = open(prefix+'/processed/' + drug_pkl, 'rb')
    drug_pth = pkl.load(drug_file)
    drug_file.close()
    disease_file = open(prefix+'/processed/' + disease_pkl, 'rb')
    disease_pth = pkl.load(disease_file)
    disease_file.close()
    train_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, train_data[:, :2], train_data[:, 2])
    eval_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, eval_data[:, :2], eval_data[:, 2])
    test_set = MyDataset(drug_pth, disease_pth, drug_metas, disease_metas, test_data[:, :2], test_data[:, 2])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, eval_loader, test_loader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

