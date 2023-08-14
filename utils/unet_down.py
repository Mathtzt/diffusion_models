import torch
import torch.nn as nn

from .residual_conv_block import ResidualConvBlock

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        
        # Cria uma lista de camadas para o bloco de downsampling
        # Cada bloco consiste de duas camadas ResidualConvBlock, seguido de uma camada MaxPool2d para downsampling
        layers = [ResidualConvBlock(in_channels, out_channels), 
                  ResidualConvBlock(out_channels, out_channels), 
                  nn.MaxPool2d(2)]
        
        # Utiliza as camadas para criar um modelo sequential do pytorch
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Passa o tensor através do modelo e retorna o output.
        return self.model(x)