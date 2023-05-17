import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,recall_score,accuracy_score, precision_score
import random
from MYMODEL import NIEE
from utils import EarlyStopping, load_bioDDG, load_bioDDG_ALL
import tools
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

seed = 1

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



num_ntype = 3
dropout_rate = 0
lr = 0.001
weight_decay = 0.0009
num_drug = 2011
num_dis = 2649
num_gene = 14177

# drug_dis_pair = 103099
# num_train = 144338 = pos + neg
# num_eval = 20619
# num_test = 41241

def run_NIEE_lp(dataset, model, hidden_dim, num_heads, num_walks, num_epochs, patience, batch_size, repeat, save_postfix):
    if dataset == 0:
        train_loader, eval_loader, test_loader = load_bioDDG(batch_size, num_walks)
    elif dataset == 2:
        train_loader, eval_loader, test_loader = load_bioDDG_ALL(batch_size, num_walks)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    auc_list = []
    ap_list = []
    for tt in range(repeat):
        if model == 0:
            net = NIEE(num_drug, num_dis, num_gene, hidden_dim, hidden_dim, hidden_dim, num_heads, dropout_rate).to(device)

        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.BCELoss()
        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True,
                                       save_path='checkpoint/checkpoint-{}_{}.pt'.format(save_postfix, tt))
        
        '''start'''
        for epoch in range(num_epochs):
            # training
            net.train()
            dur1 = []
            dur2 = []
            loss = []
            for iteration, data in enumerate(train_loader):
                # forward

                RI, IR, RIRI, IRIR, RIGI, IGIR, labels = data
                t1 = time.time()
                pred = net(RI.to(device), IR.to(device), RIRI.to(device), IRIR.to(device), RIGI.to(device), IGIR.to(device))
                train_loss = criterion(pred, labels.to(device))
                # print(type(loss))
                loss.append(train_loss)
                t2 = time.time()
                dur1.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur2.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} '.format(
                            epoch+1, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2)))
                # break
            loss = torch.mean(torch.tensor(loss))
            date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
            print('Train finish: Train_Loss {:.4f} | Time:{}'.format(loss.item(), date_str))
            # validation

            net.eval()
            val_loss = []
            t1 = time.time()
            with torch.no_grad():
                for iteration, data in enumerate(eval_loader):
                    # forward

                    RI, IR, RIRI, IRIR, RIGI, IGIR, labels = data
                    pred = net(RI.to(device), IR.to(device), RIRI.to(device), IRIR.to(device), RIGI.to(device), IGIR.to(device))
                    eval_loss = criterion(pred, labels.to(device))
                    val_loss.append(eval_loss)

                    # break
                val_loss = torch.mean(torch.tensor(val_loss))
            t2 = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch+1, val_loss.item(), t2 - t1))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
        print('Test start:  {}'.format(date_str))

        net.load_state_dict(torch.load('checkpoint/checkpoint-{}_{}.pt'.format(save_postfix, tt)))
        # net.load_state_dict(torch.load('checkpoint/best.pt'))
        # net.load_state_dict(torch.load('checkpoint/alz.pt'))
        '''end'''
        net.eval()
        test_loss = []
        y_true_test = []
        y_proba_test = []


        with torch.no_grad():
            for iteration, data in enumerate(test_loader):
                # forward

                RI, IR, RIRI, IRIR, RIGI, IGIR, labels = data
                pred = net(RI.to(device), IR.to(device), RIRI.to(device), IRIR.to(device), RIGI.to(device), IGIR.to(device))
                loss = criterion(pred, labels.to(device))
                test_loss.append(loss.item())
                if iteration == 0:
                    y_true_test=labels.squeeze(1).numpy()
                    y_proba_test=pred.squeeze(1).detach().cpu().numpy()
                    # 为了取前10大预测结果
                    drug_column = RI[:,0,0].numpy()
                    disease_column = IR[:,0,0].numpy()
                    scores = pred.squeeze(1).cpu().numpy()
                else:
                    y_true_test = np.append(y_true_test, labels.squeeze(1).numpy())
                    y_proba_test = np.append(y_proba_test, pred.squeeze(1).detach().cpu().numpy())
                    drug_column = np.append(drug_column, RI[:,0,0].numpy())
                    disease_column = np.append(disease_column, IR[:,0,0].numpy())
                    scores = np.append(scores, pred.squeeze(1).cpu().numpy())
                
                # break



        # 下面的代码为了得到alz的前20大预测药物
        # idx = scores.argsort()[-20:]
        # with open('./alz_drug.csv', 'w', encoding='utf-8') as f:
        #     for d1, d2, s in zip(drug_column[idx], disease_column[idx], scores[idx]):
        #         item = ','.join([str(d1), str(d2), str(s)])
        #         f.writelines(item + '\n')
        #         print(d1, d2, s)
        #     f.close()
        # time.sleep(1000)




        # np.savez('../HINGE/prediction_result.npz', y_true=y_true_test, y_pred=y_proba_test)
        np.savez('record/paper/{}_prediction_result-{}-{}_{}.npz'.format(save_postfix, hidden_dim,
                                                                num_heads, tt), y_true=y_true_test,y_pred=y_proba_test)
        # print(type(y_true_test))
        # print(y_true_test.shape)
        # print(type(y_proba_test))
        # print(y_proba_test.shape)
        y_pred = [1 if y_proba_test[i] >= 0.5 else 0 for i in range(len(y_proba_test))]
        auc = tools.evaluate_auc(y_proba_test, y_true_test)
        ap = tools.evaluate_ap(y_proba_test, y_true_test)
        f1score = f1_score(y_true_test, y_pred)
        accuracy = accuracy_score(y_true_test, y_pred, normalize=True)
        precision = precision_score(y_true_test, y_pred, average='binary',pos_label=1)
        recall = recall_score(y_true_test, y_pred, average='binary')
        print('Link Prediction Test')
        print('AUC = %.4f,AP = %.4f, f1-score = %.4f, accuracy= %.4f,precision =%.4f,recall =%.4f'% (auc,ap,f1score,accuracy,precision,recall))
        # print('AUC = {}'.format(auc))
        # print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)
        date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
        print(date_str) 

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))
    date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
    print(date_str)


if __name__ == '__main__':
    date_str = time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime())
    print(date_str)
    ap = argparse.ArgumentParser(description='NIEE testing for the biology dataset')
    ap.add_argument('--model', type=int, default=0,
            help='which model to used. ' +
            '0 - NIEE; ' )
    ap.add_argument('--data', type=int, default=0,
            help='which dataset to use. ' +
            # 0. train:val:test=7:1:2
            '0 - bioDDG; ' +   
            '1 - lagcn;' +
            # 2. bioDDG的所有数据均用于训练，得到的模型用来预测治疗AD的药物
            '2 - all' )
    ap.add_argument('--hidden-dim', type=int, default=128, help='Dimension of the node hidden state. Default is 128.')
    ap.add_argument('--num-heads', type=int, default=3, help='Number of the attention heads. Default is 3.')
    ap.add_argument('--num-walks', type=int, default=100, help='Number of walks per node.')
    ap.add_argument('--epoch', type=int, default=20, help='Number of epochs. Default is 20.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=128, help='Batch size. Default is 128.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='NIEE_bio',
                    help='Postfix for the saved model and result. Default is NIEE_bio.')


    args = ap.parse_args()
    run_NIEE_lp(args.data, args.model, args.hidden_dim, args.num_heads, args.num_walks, args.epoch,
                     args.patience, args.batch_size, args.repeat, args.save_postfix)
