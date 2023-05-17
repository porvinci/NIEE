import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append("..")




class IntraMetaPath(nn.Module):
    def __init__(self, in_size, out_size=128, atn_heads = 3, theta = 0.2, alpha=0.2, dropout=0.6):
        super(IntraMetaPath, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.atn_heads = atn_heads
        self.theta = theta
        self.initializer = nn.init.xavier_uniform_
        self.W = nn.Parameter(self.initializer(torch.empty(atn_heads, in_size, out_size)))
        self.a = nn.Parameter(self.initializer(torch.empty(atn_heads, out_size*2, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = dropout
        
    def forward(self, h):
        H=torch.reshape(h,(h.shape[0]*h.shape[1],h.shape[2],h.shape[3]))
        for i in range(self.atn_heads):
            Wh = torch.Tensor.matmul(H, self.W[i,:,:])
            Wh1 = torch.Tensor.matmul(Wh, self.a[i,:self.out_size,:])
            Wh2 = torch.Tensor.matmul(Wh, self.a[i,self.out_size:,:]) 
            e = self.leakyrelu(Wh1 + Wh2.permute(0,2,1))[:,0,:]
            attention = F.softmax(e).unsqueeze(1)
            attention = F.dropout(attention, self.dropout)
            h_prime = torch.Tensor.matmul(attention, Wh).squeeze(1)

            if i==0:
                Z = h_prime
            else:
                Z += h_prime
                
        Z = torch.reshape(Z,(h.shape[0],h.shape[1],self.out_size))
        Z = Z/self.atn_heads
        return Z



class InterMetaPath(nn.Module):
    def __init__(self, in_size, hidden_size=128, theta=1):
        super(InterMetaPath, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.theta=theta

    def forward(self, z):
        dim1 = z.shape[0]
        dim2 = z.shape[1]
        dim3 = z.shape[3]
        z=z.reshape(z.shape[0]*z.shape[1],z.shape[2],z.shape[3])
        w = self.project(z)
        beta = F.softmax(w,1)
        h = torch.Tensor.matmul(beta.permute(0,2,1), z).squeeze(1)
        return h.reshape(dim1, dim2, dim3)

 
class NIEE(nn.Module):
    
    def __init__(self, drug_num, dis_num, gene_num, in_size, hidden_size, out_size, num_heads, theta1=0.2, theta2=0.2):
        super(NIEE, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.drug_emb = nn.Embedding(drug_num+1, in_size, padding_idx=0)
        self.dis_emb = nn.Embedding(dis_num+1, in_size, padding_idx=0)
        self.gene_emb = nn.Embedding(gene_num+1, in_size, padding_idx=0)
        self.IntraMetaPath = nn.ModuleList()
        for i in range(0,3):  # RI，RIRI，RIGI
            self.IntraMetaPath.append(IntraMetaPath(in_size, hidden_size, num_heads, theta1))
        self.InterMetaPath = InterMetaPath(hidden_size, hidden_size, theta2)
        self.final_linear = nn.Sequential(
            nn.Linear(3*hidden_size, out_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_size, 1)
        )



    def forward(self, RI, IR, RIRI, IRIR, RIGI, IGIR):
        drug_idx = RI[:,0,0]
        dis_idx = IR[:,0,0]

        source_feature = self.drug_emb(drug_idx)
        target_feature = self.dis_emb(dis_idx)

        RI = torch.stack((self.drug_emb(RI[:,:,0]), self.dis_emb(RI[:,:,1])),2)
        IR = torch.stack((self.dis_emb(IR[:,:,0]), self.drug_emb(IR[:,:,1])),2)

        RIRI = torch.stack((self.drug_emb(RIRI[:,:,0]), self.dis_emb(RIRI[:,:,1]), self.drug_emb(RIRI[:,:,2]), self.dis_emb(RIRI[:,:,3])),2)
        IRIR = torch.stack((self.dis_emb(IRIR[:,:,0]), self.drug_emb(IRIR[:,:,1]), self.dis_emb(IRIR[:,:,2]), self.drug_emb(IRIR[:,:,3])),2)
        RIGI = torch.stack((self.drug_emb(RIGI[:,:,0]), self.dis_emb(RIGI[:,:,1]), self.gene_emb(RIGI[:,:,2]), self.dis_emb(RIGI[:,:,3])),2)
        IGIR = torch.stack((self.dis_emb(IGIR[:,:,0]), self.gene_emb(IGIR[:,:,1]), self.dis_emb(IGIR[:,:,2]), self.drug_emb(IGIR[:,:,3])),2)

        drug_features = [RI,RIRI,RIGI]
        dis_features = [IR,IRIR,IGIR]
        
        H=[]
        for i in range(0,len(drug_features)):
            h = drug_features[i]*torch.flip(dis_features[i],dims=[2])
            h = self.IntraMetaPath[i](h)
            H.append(h)
        Z = torch.stack(H,2)

        Z1 = self.InterMetaPath(Z)
        pred = Z1.sum(1)
        pred = torch.cat((source_feature,target_feature,pred),1)
        pred=self.final_linear(pred)
        pred=torch.sigmoid(pred)

        return pred