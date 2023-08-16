import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .unet import ContextUnet
from utils.diffusion_utils import DiffusionUtils as du

class DiffusionModel:
    def __init__(self,
                 timesteps: int = 500,
                 beta1: float = 1e-4,
                 beta2: float = 0.02,
                 n_feat: int = 64,       # features de dimensão oculta
                 n_cfeat: int = 5,       # context vector is of size 5
                 height: int = 16,       # imagem 16 x 16
                 ### Hiperparâmetros gerais ###
                 batch_size: int = 100,
                 n_epoch: int = 32,
                 lrate: int = 1e-3,
                 ### Diretório
                 save_dir: str = './results/'):
        
        self.timesteps = timesteps
        self.beta1 = beta1
        self.beta2 = beta2
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.height = height
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lrate = lrate

        self.a_t = None
        self.b_t = None
        self.ab_t = None

        self.model = None
        self.optimizer = None

        self.save_dir = save_dir
        self.device = du.get_device_available()

    def save(self, dirpath: str, filename: str = 'model_gan', is_torchscript: bool = False):
        if is_torchscript:
            model_scripted = torch.jit.script(self.model) # Exportando para TorchScript
            model_scripted.save(f'{dirpath}/{filename}.pt')
        else:
            torch.save(self.model.state_dict(), f'{dirpath}/{filename}.pth')

    def load(self, model_path: str, model_name: str, is_torchscript: bool = False) -> ContextUnet:
        if is_torchscript:
            return torch.jit.load(f"{model_path}/{model_name}.pt") # Ex.: model_scripted.pt
        else:
            return self.model.load_state_dict(torch.load(f"{model_path}/{model_name}.pth", map_location = self.device))

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
    
    def init_model(self, in_channels: int = 3):
        self.model = ContextUnet(in_channels = in_channels, 
                                 n_feat = self.n_feat, 
                                 n_cfeat = self.n_cfeat, 
                                 height = self.height).to(self.device)
        
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lrate)
        
    def train(self, dataloader = None, optimizer = None):
        ### Criando folders para registro ###
        models_dpath = du.create_folder(path = self.save_dir, name = 'models')
        exp_dpath = du.create_folder(path = models_dpath, name = 'ddpm_models', use_date = True)
        ### Viabilizando possibilidade de receber um otimizador na chamada ###
        if optimizer is None:
            self.init_optimizer()
        else: self.optimizer = optimizer
        ### Inicializando ruído ###
        self.set_ddpm_noise()
        ### Colocando modelo em modo de treino ###
        self.model.train()

        for epoch in range(self.n_epoch):
            print(f'######### epoch {epoch + 1} #########')
            
            # Inserindo taxa de aprendizagem com decaimento linear
            self.optimizer.param_groups[0]['lr'] = self.lrate * (1 - epoch / self.n_epoch)
            
            pbar = tqdm(dataloader, mininterval = 2)
            for x, _ in pbar:   # x: imagens
                self.optimizer.zero_grad() # zerando os gradientes
                # Alocando no device disponível
                x = x.to(self.device)
                # Perturbando os dados de entrada com ruído dado um timestep
                noise = torch.randn_like(x)
                t = torch.randint(1, self.timesteps + 1, (x.shape[0], )).to(self.device) 
                x_pert = du.perturb_input(x, t, self.ab_t, noise)
                # Predizendo o ruído que foi gerado
                pred_noise = self.model(x_pert, t / self.timesteps)
                # Calculando a loss do modelo utilizando mse entre o ruído predito e o verdadeiro
                loss = F.mse_loss(pred_noise, noise)
                # print(f"Noise loss = {loss.cpu()}")
                loss.backward()
                ## bp ##
                self.optimizer.step()

            print(f"Train Loss: {loss:.5f}")
            ### Salvando o modelo periodicamente para verificar evolução ###
            if epoch % 4 == 0 or epoch == int(self.n_epoch - 1):
                du.save(dirpath = exp_dpath, filename = f"model_{epoch}")
                # print('saved model at ' + exp_dpath + f"model_{epoch}")
    
    @torch.no_grad()
    def sampling_ddpm(self, n_sample, save_rate=20, init_model: bool = False):
        # Instanciando modelo
        if init_model:
            self.init_model()
        #####################################
        #Descomentar para caso deseje utilizar o modelo de exemplo
        # self.model.load_state_dict(torch.load(f"./_support/weights/model_trained.pth", map_location=self.device))
        # self.model.eval()
        #####################################
        self.model.eval()
        # Configurando ruído
        self.set_ddpm_noise()
        
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
    
    def visualize_samples(self, filename: str = "animation"):

        plt.clf()
        _, intermediate_ddpm = self.sampling_ddpm(n_sample = 32)
        du.plot_sample(intermediate_ddpm, 32, 4, "./results/videos_and_imgs/", filename, None, save = True)
        
        # HTML(animation_ddpm.to_jshtml())