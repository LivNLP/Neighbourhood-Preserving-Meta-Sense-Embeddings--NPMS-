import os
import time
from os.path import exists

import numpy
import torch
import  numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import logging

def cos_dis(a:torch.Tensor,b):
    assert a.shape == b.shape
    return -torch.dot(a,b)/(torch.norm(a,2)*torch.norm(b,2))

if __name__=='__main__':

    if torch.cuda.is_available():
        print('device num', torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print('current idx', idx)
        print(torch.cuda.get_device_name(idx))
    device = torch.device('cuda')

    loader1_path = '../sensembert/sensembert_norm.npz'
    loader1 = np.load(loader1_path)
    loader2_path = '../emb_npz/ares.npz'
    loader2 = np.load(loader2_path)
    vect1 = torch.tensor(loader1['vectors'],requires_grad=False,device=device)
    vect2 = torch.tensor(loader2['vectors'], requires_grad=False, device=device)
    dict1 = {k:v for k, v in zip(loader1['labels'], vect1)}
    dict2 = {k:v for k, v in zip(loader2['labels'], vect2)}
    
    #supervised part
    sup_loader = np.load("../train.npz")
    vect3 = torch.tensor(sup_loader['vectors'], requires_grad=False, device=device)
    dict3 = {k: v for k, v in zip(sup_loader['labels'], vect3)}
    
    # init projection matrices and bias
    vec2_mat = torch.eye(2048)
    vec2_mat = vec2_mat.to(device)
    vec2_mat.requires_grad = True
    vec1_mat = torch.eye(2048)
    vec1_mat = vec1_mat.to(device)
    vec1_mat.requires_grad = True
    bias = torch.zeros(2048)
    bias = bias.to(device)
    bias.requires_grad=True
    optmizer = optim.Adam([vec2_mat, vec1_mat,bias], lr=0.001)
    res = torch.zeros(1)
    res.to(device)
    UPDATE_FREQ = 6000
    SAVE_FREQ = 100000
    
    for epoch in range(10):
        cum_loss = 0
        loss = torch.zeros(1).to(device)
        tpoch = tqdm(enumerate(dict1.keys()))
        optmizer.zero_grad()
        for id,senseid in tpoch:
            res = torch.matmul(dict1[senseid], vec1_mat) + torch.matmul(dict2[senseid], vec2_mat)+bias
            # distance to source embedidng, unsupervised part
            dist = cos_dis(res,dict1[senseid])+cos_dis(res,dict2[senseid])
            # supervised part
            if senseid in dict3.keys():
                dist = dist+cos_dis(res,dict3[senseid])
            loss = loss + dist
            tpoch.set_postfix(loss=loss.data,ep=epoch,cum_loss=cum_loss)
            if (id+1)%UPDATE_FREQ==0:
                loss.backward()
                optmizer.step()
                optmizer.zero_grad()
                cum_loss = cum_loss+loss.item()
                loss = torch.zeros(1).to(device)
            if (id+1)%SAVE_FREQ==0:
                del res
                out = []
                for key in dict2.keys():
                    if key in dict1.keys():
                        temp = torch.matmul(dict2[key], vec2_mat) + torch.matmul(dict1[key], vec1_mat)+ bias
                        out.append(temp.cpu().detach().tolist())
                    else:
                        temp = torch.matmul(dict2[key], vec2_mat)
                        out.append(temp.cpu().detach().tolist())

                np.savez(f'temp/sup_ares_sensem_bias{epoch}.npz',vectors=numpy.array(out),labels =list(dict2.keys())
                         ,mat1 = vec1_mat.cpu().detach().numpy(),mat2 = vec2_mat.cpu().detach().numpy(),
                         bias = bias.cpu().detach().numpy(),
                         desc = 'mat1 is the transformation matrix for the vector which is to be aligned')
        print(cum_loss)
