# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .blocks_basic import *


class Ghostv1(nn.Module):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'Ghostv1',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn': bottleneck_channels,
            'act': activation (default=relu),
        }
        :param NAS_mode:
        '''

        super().__init__()

        if 'class' in structure_info:
            assert structure_info['class'] == self.__class__.__name__

        self.in_channels = structure_info['in']
        self.out_channels = structure_info['out']
        self.kernel_size = structure_info['k']
        self.stride = 1 if 's' not in structure_info else structure_info['s']
        self.bottleneck_channels = structure_info['btn']
        assert self.stride == 1 or self.stride == 2
        if "act" not in structure_info:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(structure_info['act']) 
        self.no_create = no_create
        self.dropout_channel = dropout_channel
        self.dropout_layer = dropout_layer

        if 'force_resproj' in structure_info:
            self.force_resproj = structure_info['force_resproj']
        else:
            self.force_resproj = False

        if 'g' in structure_info:
            self.groups = structure_info['g']
        else:
            self.groups = 1

        if 'p' in structure_info:
            self.padding = structure_info['p']
        else:
            self.padding = (self.kernel_size - 1) // 2

        self.model_size = 0.0
        self.flops = 0.0

        self.block_list = []

        ghost1_info = {
            'in': self.in_channels,
            'out': self.bottleneck_channels,
        }
        ghost2_info = {
            'in': self.bottleneck_channels,
            'out': self.out_channels,
        }
        if self.stride>1:
            conv1_info= {
                'in':self.bottleneck_channels,
                'out':self.bottleneck_channels,
                'k': 3, 
                's':self.stride, 
                'p':1, 
                'g':self.bottleneck_channels,
            }

        #ghost1
        self.ghost1 = GhostModule(ghost1_info, relu=True, **kwargs)
        if self.stride > 1:
            self.conv1 = ConvKXBN(conv1_info, **kwargs)
        #ghost2
        self.ghost2 = GhostModule(ghost2_info,relu=False, **kwargs)
        self.block_list.append(self.ghost1)
        if self.stride>1:
            self.block_list.append(self.conv1)
            self.model_size = self.model_size + self.ghost1.get_model_size() + self.ghost2.get_model_size() + self.conv1.get_model_size()
            self.flops = self.flops + self.ghost1.get_flops(1.0) + self.ghost2.get_flops(1.0/self.stride) + self.conv1.get_flops(1.0)

        else:
            self.model_size = self.model_size + self.ghost1.get_model_size() + self.ghost2.get_model_size()
            self.flops = self.flops + self.ghost1.get_flops(1.0) + self.ghost2.get_flops(1.0)
        self.block_list.append(self.ghost2)

        # shortcut
        if (self.in_channels == self.out_channels and self.stride == 1):
            self.shortcut = False #nn.Identity() False
        else:
            self.shortcut = True
            conv2_info= {
                'in':self.in_channels,
                'out':self.in_channels,
                'k': 3, 
                's':self.stride, 
                'p':1, 
                'g':self.in_channels,
            }
            conv3_info= {
                'in':self.in_channels,
                'out':self.out_channels,
                'k': 1, 
                's':1, 
                'p':0, 
            }
            self.conv2=ConvKXBN(conv2_info, no_create=no_create, **kwargs)
            self.conv3=ConvKXBN(conv3_info, no_create=no_create, **kwargs)
            # self.shortcut = nn.Sequential(
            #     self.conv2,self.conv3
            # )
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(self.in_channels, self.in_channels, 3, stride=self.stride,
            #            padding=1, groups=self.in_channels, bias=False),
            #     nn.BatchNorm2d(self.in_channels),
            #     nn.Conv2d(self.in_channels, self.out_channels, 1, stride=1, padding=0, bias=False),
            #     nn.BatchNorm2d(self.out_channels),
            # )
            self.model_size = self.model_size + self.conv2.get_model_size() + self.conv3.get_model_size()
            self.flops = self.flops +self.conv2.get_flops(1.0) + self.conv3.get_flops(1.0 / self.stride) 

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        output = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            output = self.conv1(output)

        # 2nd ghost bottleneck
        output = self.ghost2(output)

        if self.shortcut:
            residual = self.conv3(self.conv2(residual))
            
        # output += self.shortcut(residual)
        output += residual
        
        return output
        

    def get_model_size(self):
        return self.model_size


    def get_flops(self, resolution):
        return self.flops * resolution**2


    def get_num_layers(self):
        if self.stride>1:
            return 5
        else:
            return 4
        

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution
        
        
    def get_params_for_trt(self, input_resolution):
        # generate the params for yukai's predictor
        params = []
        the_res = input_resolution
        for idx, block in enumerate(self.block_list):
            if self.shortcut and idx == len(self.block_list) - 1:
                params_temp = block.get_params_for_trt(
                    the_res, elmtfused=1)
            else:
                params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        if (self.in_channels == self.out_channels and self.stride == 1):
            self.shortcut = nn.Identity()
        else:
            params_temp = self.conv2.get_params_for_trt(the_res) + self.conv3.get_params_for_trt(the_res)
        params += params_temp
        return params


    def entropy_forward(self, x, skip_relu=True, skip_bn=True, **kwarg):
        output = x
        output_std_list = []
        output_std_block = 1.0
        for the_block in self.block_list:
            output = the_block(output, skip_bn=skip_bn)
            if not skip_relu: output = self.activation_function(output)
            # print("output std: mean %.4f, std %.4f, max %.4f, min %.4f\n"%(
                    # output.mean().item(), output.std().item(), output.max().item(), output.min().item()))
            output_std_block *= output.std()
            output = output/output.std()
        output_std_list.append(output_std_block)
        return output, output_std_list


    def get_num_channels_list(self):
        if self.stride>1:
            return [self.bottleneck_channels, self.bottleneck_channels, self.out_channels]
        else:
            return [self.bottleneck_channels, self.out_channels]
        # return [self.bottleneck_channels, self.bottleneck_channels, self.out_channels]


class SuperGhostv1(BaseSuperBlock):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResConvK1KX',
            'in': in_channels,
            'out': out_channels,
            's': stride (default=1),
            'k': kernel_size,
            'p': padding (default=(k-1)//2,
            'g': grouping (default=1),
            'btn':, bottleneck_channels,
            'L': num_inner_layers,
        }
        :param NAS_mode:
        '''
        structure_info['inner_class'] = 'Ghostv1'
        super().__init__(structure_info=structure_info, no_create=no_create, inner_class=Ghostv1,
                         dropout_channel=dropout_channel, dropout_layer=dropout_layer,
                         **kwargs)


__module_blocks__ = {
    'Ghostv1': Ghostv1,
    'SuperGhostv1': SuperGhostv1,
}