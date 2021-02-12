#Hope to contain only one function  here ..which will return the base_module ( nn.Module object)

#Define the backbone network for SSD architecture ... on each of the returned tensor a classification and regression subnet is attached...according to the SSD
#architecture



#NOTE:  only need to change this function/architecture ..when using a different backbone ...meaning u can try different architectures .


import torch
import torch.nn as nn
import torch.nn.functional as F

#either create a new  network from scratch  or use a pretrained model...just remember to return the list of tensors.(which are actually feature maps onto which a regres
# sion and classification  subnet is attached.)

#NOTE: Try to make sure that the feature maps are square.....i think this can be made sure by taking INPUT images as Square.

#########################################################################################################################################################################3
#block to reduce the size of the input by half

def down_sample_blk(in_planes,out_planes):
    seq=nn.Sequential()
    
    seq.add_module("conv_1",nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=3,padding=1))
    seq.add_module("batch_norm_1",nn.BatchNorm2d(num_features=out_planes))
    seq.add_module("relu_1",nn.ReLU())
    
    seq.add_module("conv_2",nn.Conv2d(in_channels=out_planes,out_channels=out_planes,kernel_size=3,padding=1))
    seq.add_module("batch_norm_2",nn.BatchNorm2d(num_features=out_planes))
    seq.add_module("relu_2",nn.ReLU())
    
    seq.add_module("max_pool_1",nn.MaxPool2d(kernel_size=2))
    
    return seq

class base_net(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        
        
        filters=[in_channels,16,32,64,128,128,128]  #one extra for input image channels
        
        #set 6 down_sample_blocks as attributes
        for i in range(6):
            setattr(self, 'blk_%d' % i, down_sample_blk(in_planes=filters[i],out_planes=filters[i+1]))
                    
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        
                    
    def forward(self,x):
        outputs=[]
        
        
        for i in range(6):
            x=getattr(self,'blk_%d' %i)(x)
            
            if i>=2:
                outputs.append(x)
                
                
        x=self.max_pool(x)
        outputs.append(x)
        
        return outputs
            
        
        

#NOTE:  This function should be present in each custom/pretrained   .py file  and only this function will be called
# from the training.py file ....will return the nn.Module object...which on passing an image through  must return the list
#of feature_maps:

#This module is also called by the Config file...to know the shape of feature maps...by passing a  dummy tensor..and later will be deleted.

#in_channels is the number of channels present in the input_image by default it is 3  RGB channels


def backbone_network(in_channels=3):
    return base_net(in_channels)





#######################################################################################################################################################################


    




