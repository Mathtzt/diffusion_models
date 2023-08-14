import torch.nn as nn

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Esta classe define uma rede neural genérica de uma camada de feed-forward para incorporar dados de entrada de dimensionalidade input_dim num espaço de incorporação de dimensionalidade emb_dim.
        """
        
        self.input_dim = input_dim
        
        # Define as camadas da rede neural
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        # Cria um pytorch sequencial model a partir das camadas definidas
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten do tensor de entrada
        x = x.view(-1, self.input_dim)
        # Aplica as camadas do modelo para o tensor flattened

        return self.model(x)