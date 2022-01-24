# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:45:16 2021

@author: Justin
"""
import numpy as np

from tqdm import tqdm

import torchaudio

import copy
import time
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW,SGD
import torch.utils.data as Data

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from models.disc_with_confidence import Disc
from torch_datasets.mydataset import testdataset2
from utils.utils import encode_onehot
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  #to accelerate


#####################
###data processing###
#####################
torch.manual_seed(40)

#raw data's path
raw_data_dir = '../data/raw'

# what window length to use
window_length = 1.5 # seconds
print('\nTraining using window length of {} seconds...'.format(window_length))

# what sampling frequency to resample windows to
sr = 16000 # Hz

# initialize loss function 
loss_func = nn.NLLLoss().cuda()

# initialize log operator for Logarithmic Mel-scale Spectrogram
log = torchaudio.transforms.AmplitudeToDB().to(device)

# initialize Mel-scale Spectrogram operator for Logarithmic Mel-scale Spectrogram
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate = sr,
                                                n_fft = 1024,
                                                n_mels = 128,
                                                hop_length = 64).to(device)

# initialize discriminator network
FENet_param_path = '../parameters/FENet/FENet.pkl'
net = Disc(FENet_param_path).to(device)

# initialize optimizer
optimizer = torch.optim.AdamW(params = net.parameters(),
                             lr = 0.01)

# number of epoch to train and validate for
num_epochs = 30

# where to save net parameters in a .pt file
pt_filename = '{}s_{}Hz_{}epochs_with_confidence.pt'.format(str(window_length).replace('.','-'),
                                            sr,num_epochs)
param_path = '../parameters/disc/' + pt_filename

# initialize training and validation dataloaders
dataloaders = {}

batch_size = 128
                             

train_dataset = testdataset2(raw_data_dir,window_length,sr,'train',
                            only_speech = False)
    
train_dataloader = DataLoader(
                               dataset = train_dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)

val_dataset = testdataset2(raw_data_dir,window_length,sr,'val',
                            only_speech = False)
    
val_dataloader = DataLoader(
                               dataset = val_dataset,
                               batch_size = batch_size,
                               shuffle = True,
                               pin_memory=True,
                               num_workers=0)


# record the best validation loss across epochs
best_val_loss = 1e10

print("train_loader的batch数量为",len(train_dataloader))

cudnn.benchmark = True



####################################
###begin trainning and validating###
####################################



###想法是二分类，每一行输出两个值，但有分布外的数据集，我们加上信任度做判断。

def val(loader): #val（val_loader)
    net.eval()    # Change model to 'eval' mode 
    correct = []
    confidence = []
    probability = []

    for x, labels in loader:
        
        x = Variable(x, volatile=True).cuda()
        labels = labels.cuda().type_as(x)
        
        log_mel_spec = log(mel_spec(x))
        
        pred,conf=net(log_mel_spec)
        
        pred = F.softmax(pred, dim=-1)
        conf = F.sigmoid(conf).data.view(-1)
        
        
        pred_value, pred = torch.max(pred.data, 1)   #每行的最大值，及最大值的位置，输出为两个行张量
        correct.extend((pred == labels).cpu().numpy()) #添加并转变为numpy数组
        probability.extend(pred_value.cpu().numpy())
        confidence.extend(conf.cpu().numpy())


    correct = np.array(correct).astype(bool)
    probability = np.array(probability)
    confidence = np.array(confidence)


    val_acc = np.mean(correct)  #分类准确率
    conf_min = np.min(confidence)   #最小自信度
    conf_max = np.max(confidence)   #最大自信度
    conf_avg = np.mean(confidence)  #自信度的平均值

    net.train()
    
    return val_acc, conf_min, conf_max, conf_avg



# Start with a reasonable guess for lambda
lmbda = 0.1


    
# #begin trainning
for epoch in range(30):

    nll_loss_avg = 0.
    confidence_loss_avg = 0.
    correct_count = 0.
    total = 0.

    progress_bar = tqdm(train_dataloader)   #tqdm显示训练的进度条
    for i, (x, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        x = x.cuda()
        labels = labels.cuda()
        
        log_mel_spec = log(mel_spec(x))
        
        labels_onehot = encode_onehot(labels, 2)  #将labels变为独热编码，一个label占一行

        net.zero_grad()
        
        pred_original,confidence=net(log_mel_spec)

        pred_original = F.softmax(pred_original, dim=-1)
        confidence = F.sigmoid(confidence)

        eps = 1e-12
        pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
        confidence = torch.clamp(confidence, 0. + eps, 1. - eps)


        ### Randomly set half of the confidences to 1 (i.e. no hints)
        b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).cuda()
        conf = confidence * b + (1 - b)
        pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
        pred_new = torch.log(pred_new).cuda()
        
        nll_loss = loss_func(pred_new, labels)
        confidence_loss = torch.mean(-torch.log(confidence))

        total_loss = nll_loss + (lmbda * confidence_loss)
  
        if 0.3 > confidence_loss:
            lmbda = lmbda / 1.01
        elif 0.3 <= confidence_loss:
            lmbda = lmbda / 0.99


        total_loss.backward()
        optimizer.step()

        nll_loss_avg += nll_loss
        confidence_loss_avg += confidence_loss

        pred_idx = torch.max(pred_original.data, 1)[1]
        total += labels.size(0)
        correct_count += (pred_idx == labels.data).sum() #预测正确的个数
        accuracy = correct_count / total     #预测准确率

        progress_bar.set_postfix(
            nll='%.3f' % (nll_loss_avg / (i + 1)),  #运行到第i+1个batch的平均
            confidence_loss='%.3f' % (confidence_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
        
        # torch.save(net.state_dict(), 'E:/迅雷下载/Cough-Detection-disc/parameters/disc/1-5s_16000Hz_5epochswith_confidence_budget_0.3.pt')
        
        
        



    test_acc, conf_min, conf_max, conf_avg = val(val_dataloader)      #在测试集测试
    tqdm.write('test_acc: %.3f, conf_min: %.3f, conf_max: %.3f, conf_avg: %.3f' % (test_acc, conf_min, conf_max, conf_avg))

    print('epoch:', str(epoch), 'train_acc:', str(accuracy), 'test_acc:' ,str(test_acc))

    torch.save(net.state_dict(), '../parameters/disc/1-5s_16000Hz_5epochswith_confidence_budget_0.3.pt')
    
    
    
        

        














































































