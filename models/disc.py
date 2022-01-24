import torch
"""
if running this file independently, then use:

from FENet import FENet

else if importing this module from another file, then use:
    
from .FENet import FENet
"""
from .FENet import FENet

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
        
        self.fc1 = torch.nn.Linear(
            in_features = self.feature_extractor.stage7[0].out_channels,
            out_features = 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    FENet_param_path = '../parameters/FENet/FENet.pkl'
    net = Disc(FENet_param_path)
    # minimum input size is 128 x 128
    x = torch.randn(8,1,128,128)
    y = net(x)