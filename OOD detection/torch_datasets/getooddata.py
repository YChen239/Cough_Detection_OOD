# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:47:44 2021

@author: Justin
"""

import pathlib
import torch
import torchaudio
import pandas as pd



class getesc50(torch.utils.data.Dataset):
    
    def __init__(self,sample_rate):
        
        self.raw_data_dir = '../data/raw'   #raw data path
    
        csv_path = '../data/1-5s/ESC50.csv'  #csv path
        
        self.metadata = pd.read_csv(csv_path,header = None) #the csv file
        
        self.new_sr = sample_rate  #new sample rate
    
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self,idx):   #get item by index
        
        path = str(self.raw_data_dir) + '/' + str(self.metadata.iloc[idx,0])
        
        start_sec = self.metadata.iloc[idx,1] #start time
        end_sec = self.metadata.iloc[idx,2]   #end time
        
        path=str(path)
        data,sr=torchaudio.load(filepath = path) #find out original sample rate

        start = round(sr * start_sec)
        length = round(sr * (end_sec - start_sec))
        x = torchaudio.load(filepath = path,frame_offset = start,num_frames = length)[0]
    
        x = torchaudio.transforms.Resample(sr,self.new_sr)(x)#transform the sr
        x = torch.mean(x,dim=0,keepdim=True)
    
        return x