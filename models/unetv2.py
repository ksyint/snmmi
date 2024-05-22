"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
# from torchsummary import summary
import torch
import time



class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

    
    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class UNet3DV2(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[128, 256, 512, 1024], bottleneck_channel=2048) -> None:
        super(UNet3DV2, self).__init__()
        
        # Analysis path
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_channels[0])
        self.a_block2 = Conv3DBlock(in_channels=level_channels[0], out_channels=level_channels[1])
        self.a_block3 = Conv3DBlock(in_channels=level_channels[1], out_channels=level_channels[2])
        self.a_block4 = Conv3DBlock(in_channels=level_channels[2], out_channels=level_channels[3])

        # Bottleneck
        self.bottleneck = Conv3DBlock(in_channels=level_channels[3], out_channels=bottleneck_channel, bottleneck=True)

        # Synthesis path
        self.s_block4 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_channels[3])
        self.s_block3 = UpConv3DBlock(in_channels=level_channels[3], res_channels=level_channels[2])
        self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], res_channels=level_channels[1])
        self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], res_channels=level_channels[0], last_layer=True, num_classes=num_classes)

    def forward(self, input):
        # Analysis path forward feed
        out, residual1 = self.a_block1(input)
        out, residual2 = self.a_block2(out)
        out, residual3 = self.a_block3(out)
        out, residual4 = self.a_block4(out)

        # Bottleneck
        latent, _ = self.bottleneck(out)

        # Synthesis path forward feed
        out = self.s_block4(latent, residual4)
        out = self.s_block3(out, residual3)
        out = self.s_block2(out, residual2)
        out = self.s_block1(out, residual1)

        return out, latent