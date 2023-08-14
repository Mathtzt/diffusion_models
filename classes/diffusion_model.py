import numpy as np
import matplotlib.pyplot as plt
import torch

from .unet import ContextUnet
from utils.diffusion_utils import DiffusionUtils as du

from IPython.display import HTML

class DiffusionModel:
    def __init__(self,
                 timesteps: int = 500,
                 beta1: float = 1e-4,
                 beta2: float = 0.02,
                 n_feat: int = 64,       # features de dimensão oculta
                 n_cfeat: int = 5,       # context vector is of size 5
                 height: int = 16,       # imagem 16 x 16
                 save_dir: str = './results/models/'):
        
        self.timesteps = timesteps
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.height = height

        self.a_t = None
        self.b_t = None
        self.ab_t = None

        self.model = None
        self.save_dir = save_dir
        self.device = du.get_device_available()

    def set_ddpm_noise(self):
        self.b_t = (self.beta2 - self.beta1) * torch.linspace(0, 1, self.timesteps + 1, device = self.device) + self.beta1
        self.a_t = 1 - self.b_t
        self.ab_t = torch.cumsum(self.a_t.log(), dim=0).exp()    
        self.ab_t[0] = 1

    def denoise_add_noise(self, x, t, pred_noise, z = None):
        # Remover o ruído predito e adicionar rúido de volta para evitar que colapse o treinamento
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt()
        
        return mean + noise
    
    @torch.no_grad()
    def sampling_ddpm(self, n_sample, save_rate=20):
        # Instanciando modelo
        self.model = ContextUnet(in_channels = 3, 
                                 n_feat = self.n_feat, 
                                 n_cfeat = self.n_cfeat, 
                                 height = self.height).to(self.device)
        
        # x_T ~ N(0, 1), ruído inicial
        samples = torch.randn(n_sample, 3, self.height, self.height).to(self.device)  

        # Array para registrar os steps gerados
        intermediate = [] 
        for i in range(self.timesteps, 0, -1):
            print(f'Amostrando timestep {i:3d}', end='\r')

            # Reshape time tensor
            t = torch.tensor([i / self.timesteps])[:, None, None, None].to(self.device)

            # Amostragem de um ruído aleatório para injetar de volta. Não adiciona ruído em i = 1.
            z = torch.randn_like(samples) if i > 1 else 0

            eps = self.model(samples, t) # ruído predito e_(x_t,t)
            samples = self.denoise_add_noise(samples, i, eps, z)
            if i % save_rate == 0 or i == self.timesteps or i < 8:
                intermediate.append(samples.detach().cpu().numpy())

        intermediate = np.stack(intermediate)
        
        return samples, intermediate
    
    def visualize_samples(self):

        plt.clf()
        _, intermediate_ddpm = self.sampling_ddpm(n_sample = 32)
        animation_ddpm = du.plot_sample(intermediate_ddpm, 32, 4, "./results/videos_and_imgs/", "ani_run", None, save = True)
        
        HTML(animation_ddpm.to_jshtml())