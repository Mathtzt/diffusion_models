import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()

        # Verificar se os canais de entrada e saída são os mesmos para a conexão residual
        self.same_channels = in_channels == out_channels

        # Flag que indica se a conexão residual deve ou não ser utilizada
        self.is_res = is_res

        # Primeira camada convolucional
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   # 3x3 kernel | stride 1 | padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Função de ativação: GELU 
        )

        # Segunda camada convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel | stride 1 | padding 1
            nn.BatchNorm2d(out_channels),   # Batch normalization
            nn.GELU(),   # Função de ativação: GELU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Se usar conexão residual
        if self.is_res:
            # Aplicando a primeira camada convolucional
            x1 = self.conv1(x)

            # Aplicando a segunda camada convolucional
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            # Se os canais de entrada e saída são os mesmo, adicional conexão residual diretamente
            if self.same_channels:
                out = x + x2
            else:
                # Se não, aplicar uma camada convolucional 1x1 para ajustar as dimensões antes de adicionar a conexão residual
                shortcut = nn.Conv2d(in_channels = x.shape[1], 
                                     out_channels = x2.shape[1], 
                                     kernel_size=1, 
                                     stride=1, 
                                     padding=0).to(x.device)
                
                out = shortcut(x) + x2
            #print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normaliando saída do tensor
            return out / 1.414 # TODO entender o porque de ser 1.414

        # Se não estiver usando conexão residual, retornar a saída da segunda camada convolucional
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            return x2

    # Método para obter o número de canais da saída para esse bloco
    def get_out_channels(self):
        
        return self.conv2[0].out_channels

    # Método para definir o número de canais de saída para esse bloco
    def set_out_channels(self, out_channels):

        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels