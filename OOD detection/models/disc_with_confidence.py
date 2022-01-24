# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:40:58 2021

@author: Justin
"""


import copy
import torch
import torch.nn as nn



###########################
###construction of FENet###
###########################

class FENet(nn.Module):
    
    def __init__(self,param_path):
        
        # run nn.Module's constructor
        
        super(FENet,self).__init__()
        
        # path to .pkl file containing the pre-trained parameters
        
        self.param_path = param_path
        
        # build the net
        
        self.build_net()
        
        # load the pre-trained parameters
        
        self._load_parameters()
        
        # need this in case log Mel-scale spectrogram input is smaller
        # than 200 x 200
        
        self.interp = torch.nn.Upsample(size = (200,200),
                                        mode = 'bicubic')
    
    def build_net(self):
  
        in_channels = 16
        
        conv1 = nn.Conv2d(in_channels = 1,
                          out_channels = in_channels,
                          kernel_size = (3,3),
                          padding = (1,1))
 
        batch_norm1 = nn.BatchNorm2d(num_features = 16)
        
        activation = nn.ReLU()
        
        conv2 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels,
                          kernel_size = (3,3),
                          padding = (1,1))
        
        batch_norm2 = nn.BatchNorm2d(num_features = 16)
        
        pooling = nn.MaxPool2d(kernel_size = (2,2))
        
        # first stage
        
        stages = [nn.Sequential(conv1,
                                batch_norm1,
                                activation,
                                conv2,
                                batch_norm2,
                                activation,
                                pooling)]
        
        # next 4 stages
        
        for i in range(4):
            
            conv1 = nn.Conv2d(in_channels = in_channels,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            batch_norm1 = nn.BatchNorm2d(num_features = in_channels * 2)
            
            conv2 = nn.Conv2d(in_channels = in_channels * 2,
                              out_channels = in_channels * 2,
                              kernel_size = (3,3),
                              padding = (1,1))
            
            batch_norm2 = nn.BatchNorm2d(num_features = in_channels * 2)
            
            stages += [nn.Sequential(conv1,
                                     batch_norm1,
                                     activation,
                                     conv2,
                                     batch_norm2,
                                     activation,
                                     pooling)]
            
            in_channels = in_channels * 2
            
        # 6th stage, in_channels = 256
        
        conv1 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels * 2,
                          kernel_size = (3,3),
                          padding = (1,1))
            
        batch_norm = nn.BatchNorm2d(num_features = in_channels * 2)
            
        stages += [nn.Sequential(conv1,
                                 batch_norm,
                                 activation,
                                 pooling)]
        
        in_channels = in_channels * 2
        
        # final stage, in_channels = 512
        
        conv1 = nn.Conv2d(in_channels = in_channels,
                          out_channels = in_channels * 2,
                          kernel_size = (2,2),
                          padding = (0,0))
            
        batch_norm = nn.BatchNorm2d(num_features = in_channels * 2)
            
        stages += [nn.Sequential(conv1,
                                 batch_norm,
                                 activation)]
        
        # assign names to the stages for the state_dict
        
        self.stage1 = stages[0]
        self.stage2 = stages[1]
        self.stage3 = stages[2]
        self.stage4 = stages[3]
        self.stage5 = stages[4]
        self.stage6 = stages[5]
        self.stage7 = stages[6]
    
    def _load_parameters(self):
        
        """
        Assign parameters from .pkl file to layers.
        """
        
        # load the state dict from the .pkl file
        
        self.old_state_dict = torch.load(f = self.param_path,
                                         map_location = torch.device('cpu'))
        
        # make a copy to use load_state_dict() method later. deepcopy needed
        # because dict is a mutable object
        
        state_dict = copy.deepcopy(self.state_dict())
        
        for key,value in self.old_state_dict.items():
            
            # skip layer 19
            
            if key[12:14].isdigit() and int(key[12:14]) == 19:
                continue
            
            # get the name of the parameter in the new state dict corresponding
            # to the name of the parameter in the old state dict
            
            parameter_name = self.map_param_name(key)
            
            state_dict[parameter_name] = value
        
        # modify the net's state dict
        
        self.load_state_dict(state_dict)
        
    def map_param_name(self,key):
        
        """
        Maps names of parameters in the old state dict to names of parameters
        in the new state dict.
        """
        
        # get layer number in the old state dict
        
        if key[12:14].isdigit():
            layer_num = int(key[12:14])
        else:
            layer_num = int(key[12])
        
        # get sub-layer number in the old state dict. 0 means conv layer and
        # 1 means batch norm layer
        
        sub_layer_num = int(key.split('.')[2])
        
        # get parameter type in the old state dict. This can be 'weight' or
        # 'running_mean', for example
        
        parameter_type = key.split('.')[-1]
        
        # map layer number and sub-layer number in the old state dict to stage
        # number and sub-stage number in the new state dict. Note that only
        # convolutional layers and batch normalization layers have parameters
        
        if layer_num == 1 or layer_num == 2:
            stage = '1'
        elif layer_num == 4 or layer_num == 5:
            stage = '2'
        elif layer_num == 7 or layer_num == 8:
            stage = '3'
        elif layer_num == 10 or layer_num == 11:
            stage = '4'
        elif layer_num == 13 or layer_num == 14:
            stage = '5'
        elif layer_num == 16:
            stage = '6'
        elif layer_num == 18:
            stage = '7'
        
        sub_stage = self._get_sub_stage_number(layer_num,sub_layer_num)
        
        parameter_name = 'stage'+stage+'.'+sub_stage+'.'+parameter_type
        
        return parameter_name
        
    def _get_sub_stage_number(self,layer_num,sub_layer_num):
        
        """
        Helper function to return the sub-stage number in the new state dict
        """
        
        if layer_num < 16:
        
            # if conv and first layer
            
            if sub_layer_num == 0 and ((layer_num % 3) % 2) == 1: 
                sub_stage_num = '0'
            
            # if batch norm and first layer
            
            elif sub_layer_num == 1 and ((layer_num % 3) % 2) == 1:
                sub_stage_num = '1'
                
            # if conv and second layer
            
            elif sub_layer_num == 0 and ((layer_num % 3) % 2) == 0: 
                sub_stage_num = '3'
            
            # if batch norm and second layer
            
            else: 
                sub_stage_num = '4'
        
        else:
            
            # if conv layer
            
            if sub_layer_num == 0:
                sub_stage_num = '0'
            
            # if batch norm layer
            
            else:
                sub_stage_num = '1'
        
        return sub_stage_num
    
    def forward(self,x):
        
        if x.shape[2] < 200 or x.shape[3] < 200:
            x = self.interp(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        
        # average of all pixels in each feature map
        
        x = nn.functional.avg_pool2d(input = x,
                                     kernel_size = x.shape[2:])
        
        # flatten from N x 1024 x 1 x 1 to N x 1024, where N is the batch size
        
        x = torch.flatten(input = x,
                          start_dim = 1)
        
        return x

#test the net if necessary
# if __name__ == '__main__':
    
#     import torchaudio
    
#     param_path = '../parameters/FENet/FENet.pkl'
    
#     net = FENet(param_path)
    
#     x = torch.rand(1,1,150,200)
    
#     y = net(x)
#     print(y.shape)


########################################
###discriminator part with confidence###
########################################

class Disc(torch.nn.Module):
    
    def __init__(self,FENet_param_path):
        
        # constructor of torch.nn.Module
        
        super(Disc, self).__init__()
        
        # initialize feature extractor
        
        self.feature_extractor = FENet(FENet_param_path)
        
        # freeze parameters of feature extractor
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # initialize logistic regression
        
        self.fc1 = nn.Linear(
            in_features = self.feature_extractor.stage7[0].out_channels,
            out_features = 2)  #cough and non-cough
        
        self.conf = nn.Linear(self.feature_extractor.stage7[0].out_channels, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        pred = self.fc1(x)
        confidence=self.conf(x)
        return pred,confidence
    
    
#test the discriminator with confidence

# if __name__ == '__main__':
    
#     import torchaudio
    
#     param_path = '../parameters/FENet/FENet.pkl'
    
#     net = Disc(param_path)
    
#     x = torch.rand(3,1,150,120)
    
#     pred, confidence = net(x)
#     print(pred.shape)
#     print(confidence.shape)    #get [3,2] and [3,1]


























