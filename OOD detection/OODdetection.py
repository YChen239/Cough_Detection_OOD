# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:45:59 2021

@author: Justin
"""

import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchaudio

from torch.utils.data import Dataset

from torch_datasets.mydataset import testdataset2
from torch_datasets.getooddata import getesc50
from torch_datasets.getresp import getresp

from models.disc_with_confidence import Disc

from utils.ood_metrics import detection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

window_length = 1.5 # seconds
print('\nTesting OOD using window length of {} seconds...'.format(window_length))


FENet_param_path = '../parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)

raw_data_dir = '../data/raw'

sr = 16000

filename='1-5s_16000Hz_5epochswith_confidence_budget_0.3_epoch (19).pt'

log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).to(device)


ind_dataset=testdataset2(raw_data_dir,window_length,sr,'test',
                            only_speech = False)

ood_dataset=getresp(sr)


#dataLoader
ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset,
                                         batch_size=8,
                                         shuffle=False,
                                         
                                         num_workers=0)

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=8,
                                         shuffle=False,
                                        
                                         num_workers=0)



#载入网络参数

pretrained_dict = torch.load(str(filename))
net.load_state_dict(pretrained_dict)
net = net.cuda()

net.eval()



#输出自信度指数
def evaluate(data_loader, mode):
    out = []
    
    for data in data_loader:
        if type(data) == list:
            x, labels = data
        else:
            x = data
            
        
        x.requires_grad_()
        x = x.cuda()
        x.retain_grad()
        
        log_mel_spec = log(mel_spec(x))
        log_mel_spec.requires_grad_()
        log_mel_spec.retain_grad()
        
        if mode == 'confidence':
            _, confidence = net(log_mel_spec)
            confidence = F.sigmoid(confidence)   #列张量仍然变为列张量
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'confidence_scaling':
            epsilon = 0.001

            net.zero_grad()
            _, confidence = net(log_mel_spec)
            confidence = F.sigmoid(confidence).view(-1)   #变成一维张量
            loss = torch.mean(-torch.log(confidence))
            loss.backward()
            
            log_mel_spec = log_mel_spec - epsilon * torch.sign(log_mel_spec.grad)
            
           
            _, confidence = net(log_mel_spec)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)



    out = np.concatenate(out)
    return out


ind_scores = evaluate(ind_loader, 'confidence_scaling')
ind_labels = np.ones(ind_scores.shape[0]) ##分布内数据的标签全部为1

ood_scores = evaluate(ood_loader, 'confidence_scaling')
ood_labels = np.zeros(ood_scores.shape[0])   ##分布外数据的标签全部为0

labels = np.concatenate([ind_labels, ood_labels])
scores = np.concatenate([ind_scores, ood_scores])  ##将标签拼接，自信度拼接

ind=pd.DataFrame(ind_scores)
ind.to_csv('ind.csv')

ood=pd.DataFrame(ood_scores)
ood.to_csv('ood.csv')



detection_error, best_delta = detection(ind_scores, ood_scores)
auroc = metrics.roc_auc_score(labels, scores)
aupr_in = metrics.average_precision_score(labels, scores)
aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

#recall=TP/(TP+FN)
Y1 = ood_scores
X1 = ind_scores
recall=np.sum(np.sum(X1 > best_delta)) / (np.sum(np.sum(X1 > best_delta)) + np.sum(np.sum(X1 < best_delta))+1e-10)

#precison=TP/(TP+FP+1e-10)
precision=np.sum(np.sum(X1 > best_delta)) / (np.sum(np.sum(X1 > best_delta))+np.sum(np.sum(Y1 > best_delta))+1e-10)


#accuracy=(TP+TN)/(TP+TN+FP+FN+1e-10)



print("")
print("Method: " + 'confidence')
print("Detection error (lower is better): ", detection_error)
print("Best threshold:", best_delta)
print("AUROC (higher is better): ", auroc)
print("AUPR_IN (higher is better): ", aupr_in)
print("AUPR_OUT (higher is better): ", aupr_out)
print('recall: ',recall)
print('precision: ',precision)
print('accuracy: ',1-detection_error)



















