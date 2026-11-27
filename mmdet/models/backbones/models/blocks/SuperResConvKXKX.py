# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import os,sys
import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .blocks_basic import *


class ResConvKXKX(nn.Module):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''
        :param structure_info: {
            'class': 'ResConvKXKX',
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
            # self.activation_function = torch.relu
            self.activation_function = nn.ReLU()
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

        conv1_info = {'in': self.in_channels, 'out': self.bottleneck_channels, 'k': 3,
                            's': self.stride, 'g': self.groups, 'p': self.padding}
        conv2_info = {'in': self.bottleneck_channels, 'out': self.out_channels, 'k': 3,
                            's': 1, 'g': self.groups, 'p': self.padding}

        self.conv1 = ConvKXBN(conv1_info, no_create=no_create, **kwargs)
        self.conv2 = ConvKXBN(conv2_info, no_create=no_create, **kwargs)

        self.block_list.append(self.conv1)
        self.block_list.append(self.conv2)

        self.model_size = self.model_size + self.conv1.get_model_size() + self.conv2.get_model_size()
        self.flops = self.flops + self.conv1.get_flops(1.0) + self.conv2.get_flops(1.0) \
            + self.bottleneck_channels / self.stride ** 2 + self.out_channels / self.stride ** 2 # add relu flops

        # residual link
        if self.stride == 2:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
            self.flops = self.flops + self.in_channels
        else:
            if self.no_create:
                pass
            else:
                self.residual_downsample = nn.Identity()

        if self.in_channels != self.out_channels or self.force_resproj:
            self.residual_proj = ConvKXBN({'in': self.in_channels, 'out': self.out_channels, 'k': 1,
                                           's': 1, 'g': 1, 'p': 0}, no_create=no_create)
            self.model_size = self.model_size + self.residual_proj.get_model_size()
            self.flops = self.flops + self.residual_proj.get_flops(1.0 / self.stride)
        else:
            if self.no_create:
                pass
            else:
                self.residual_proj = nn.Identity()

    def forward(self, x, compute_reslink=True):
        reslink = self.residual_downsample(x)
        reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)
        output = self.activation_function(output)
        output = self.conv2(output)
        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        if self.dropout_layer is not None:
            if np.random.rand() <= self.dropout_layer:
                output = 0 * output + reslink
            else:
                output = output + reslink
        else:
            output = output + reslink

        if self.dropout_channel is not None:
            output = F.dropout(output, self.dropout_channel, self.training)

        output = self.activation_function(output)

        return output
        

    def get_model_size(self):
        return self.model_size


    def get_flops(self, resolution):
        return self.flops * resolution**2


    def get_num_layers(self):
        return 2
        

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
            if self.residual_proj and idx==len(self.block_list)-1:
                params_temp = block.get_params_for_trt(the_res, elmtfused=1) # if reslink, elmtfused=1 
            else:
                params_temp = block.get_params_for_trt(the_res)
            the_res = block.get_output_resolution(the_res)
            params += params_temp
        if isinstance(self.residual_proj, ConvKXBN):
            params_temp = self.residual_proj.get_params_for_trt(the_res)
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
        return [self.bottleneck_channels, self.out_channels]


class SuperResConvKXKX(BaseSuperBlock):
    def __init__(self, structure_info, no_create=False,
                 dropout_channel=None, dropout_layer=None,
                 **kwargs):
        '''

        :param structure_info: {
            'class': 'SuperResConvKXKX',
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
        structure_info['inner_class'] = 'ResConvKXKX'
        super().__init__(structure_info=structure_info, no_create=no_create, inner_class=ResConvKXKX,
                         dropout_channel=dropout_channel, dropout_layer=dropout_layer,
                         **kwargs)


__module_blocks__ = {
    'ResConvKXKX': ResConvKXKX,
    'SuperResConvKXKX': SuperResConvKXKX,
}