import torch
from torch import nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self,weights_path):
        super(VGG16,self).__init__()
        custom_weights = torch.load(weights_path)
        vgg16 = models.vgg16()
        vgg16.load_state_dict(custom_weights)
        self.features = vgg16.features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x),self.features[x])
        for x in range(4,9):
            self.slice2.add_module(str(x),self.features[x])
        for x in range(9,16):
            self.slice3.add_module(str(x),self.features[x])
        for x in range(16,23):
            self.slice4.add_module(str(x),self.features[x])
    
    def forward(self,x):
        
        x = self.slice1(x)
        feat1 = x
        x = self.slice2(x)
        feat2 = x
        x = self.slice3(x)
        feat3 = x
        x = self.slice4(x)
        feat4 = x
        
        return [feat1,feat2,feat3,feat4]
                
        