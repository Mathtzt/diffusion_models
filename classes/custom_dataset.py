import numpy as np
import torch

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, sfilename, lfilename, transform, null_context = False):
        self.sprites = np.load(sfilename)
        self.slabels = np.load(lfilename)
        print(f"sprite shape: {self.sprites.shape}")
        print(f"labels shape: {self.slabels.shape}")

        self.transform = transform
        self.null_context = null_context
        self.sprites_shape = self.sprites.shape
        self.slabel_shape = self.slabels.shape
        print(f"new sprite shape: {self.sprites.shape}")
        print(f"new labels shape: {self.slabels.shape}")
                
    # Retorna o número de imagens no dataset
    def __len__(self):
        return len(self.sprites)
    
    # Retorna a imagem e label dado um index
    def __getitem__(self, idx):
        # Retorna a imagem e label como uma tupla
        if self.transform:
            image = self.transform(self.sprites[idx])
            if self.null_context:
                label = torch.tensor(0).to(torch.int64)
            else:
                label = torch.tensor(self.slabels[idx]).to(torch.int64)
        return (image, label)

    def getshapes(self):
        # Retorna shape dos dados e labels
        return self.sprites_shape, self.slabel_shape