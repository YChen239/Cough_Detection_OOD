# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 10:31:52 2021

@author: Justin
"""
import pathlib
import torch
import torchaudio
import pandas as pd


class testdataset2(torch.utils.data.Dataset):
    
    def __init__(self,raw_data_dir,window_length,sample_rate,mode,
                  only_speech = False):
        
        self.only_speech = only_speech
        self.raw_data_dir = pathlib.Path(raw_data_dir)#raw data path
        
        csv_path = pathlib.Path(self.raw_data_dir.parent,
                                str(window_length).replace('.','-') + 's',
                                'data_'+mode+'.csv')  #get csv path
        self.metadata = pd.read_csv(csv_path,header = None) #the csv file
        
        # filter out non-speech files if necessary
        
        if only_speech:
            where = self.metadata[0].str.contains('0_LIBRISPEECH')
            self.metadata = self.metadata[where]
        else:
            labels = self.metadata[0].str.slice(0,1).astype(int)
            num_cough = sum(labels == 0)
            num_speech = sum(labels == 1)
            weights = 1. / torch.tensor([num_speech,num_cough])
            self.sample_weights = weights[torch.tensor(labels.to_list())]
               
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
        x = torchaudio.load(filepath = path,frame_offset = int(start),num_frames = int(length))[0]
    
        x = torchaudio.transforms.Resample(sr,self.new_sr)(x)#transform the sr
        x = torch.mean(x,dim=0,keepdim=True)
        
        if not self.only_speech:
            
            label = int(self.metadata.iloc[idx,0][0])
            

            return x,label
        
        else:
            return x
        

























