import torch
import torch.nn as nn

from utils.residual_conv_block import ResidualConvBlock
from utils.unet_down import UnetDown
from utils.unet_up import UnetUp
from utils.embed_fc import EmbedFC

class ContextUnet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_feat=256, 
                 n_cfeat=10, # cfeat = features de contexto
                 height=28):  
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        
        self.in_channels = in_channels  # Número de canais de entrada
        self.n_feat = n_feat            # Número de feature maps intermediários
        self.n_cfeat = n_cfeat          # Número de classes
        self.h = height                 # Assume-se h == w. Deve ser divisível por 4, então 28, 24, 20, 16... # TODO Por quê?

        # Inicializando a camada convolucional inicial
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Inicializando o caminho downsampling da rede UNet com dois níveis
        self.down1 = UnetDown(n_feat, n_feat)        # down1: [10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)    # down2: [10, 256, 4,  4]
        
        # Backup [original]: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embedding do timestep e labels de contexto com uma camada totalmente conectada da rede neural
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # Inicializando o caminho de upsampling da rede UNet com três níveis
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4), # upsampling  
            nn.GroupNorm(8, 2 * n_feat), # normalização                       
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Inicializando as últimas camadas convolucionais para mapear o mesmo número de canais da imagem de entrada
        self.out = nn.Sequential(   
            # Reduzir o número de feature maps
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1), # in_channels, out_channels, kernel_size, stride:1, padding:0 
            nn.GroupNorm(8, n_feat), # normalização
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1), # map para o mesmo número de canais como entrada
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : imagem de entrada
        t : (batch, n_cfeat)      : timestep
        c : (batch, n_classes)    : label de contexto
        """

        # Passando a imagem de entrada através da camada convolucional inicial
        x = self.init_conv(x)
        # Passando o resultada para o caminho de downsampling
        down1 = self.down1(x)       # [10, 256, 8, 8]
        down2 = self.down2(down1)   # [10, 256, 4, 4]
        
        # Convertendo os feature maps para um vetor e aplicando uma ativação
        hiddenvec = self.to_vec(down2)
        
        # mascarar o contexto se context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
            
        # Embedding o contexto e timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)   # (batch, 2 * n_feat, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        ## print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        
        up1 = self.up0(hiddenvec)
        # Adicionando e multiplicando múltiplos embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2) 
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        
        return out