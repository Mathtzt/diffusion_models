import torch
import torch.nn as nn

from .residual_conv_block import ResidualConvBlock

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        
        # Cria uma lista de camadas para o bloco de upsampling
        # O bloco consiste de uma camada ConvTranspose2d para upsampling, seguida de duas camadas ResidualConvBlock
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        # Utiliza as camadas para criar um modelo sequential do pytorch
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concate o tensor de entrada x com o tensor de ligação de deslocado/saltado ao longo da dimensão do canal
        x = torch.cat((x, skip), 1)
        
        # Passa o tensor concatenado através do modelo e retorna o output.
        x = self.model(x)
        return x