import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

class DiffusionUtils:

    @staticmethod
    def get_device_available():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def unorm(x):
        # resultado dentre [0,1]
        # x = (h, w, c) | (height, width, channel)
        xmax = x.max((0, 1))
        xmin = x.min((0, 1))

        return (x - xmin)/(xmax - xmin)
    
    @staticmethod
    def norm_all(store, n_t, n_s):
        # normalização de todas as amostras em todos os timesteps
        nstore = np.zeros_like(store)
        for t in range(n_t):
            for s in range(n_s):
                nstore[t,s] = DiffusionUtils.unorm(store[t,s])
        return nstore
    
    @staticmethod
    def norm_torch(x_all):
        # normalização de todas as amostras em todos os timesteps
        # input é: (n_samples, c, h, w) 
        x = x_all.cpu().numpy()

        xmax = x.max((2,3))
        xmin = x.min((2,3))
        xmax = np.expand_dims(xmax,(2,3)) 
        xmin = np.expand_dims(xmin,(2,3))

        nstore = (x - xmin)/(xmax - xmin)

        return torch.from_numpy(nstore)
    
    @staticmethod
    def gen_tst_context(n_cfeat):
        # Gerar vetores de contexto para teste
        # human, non-human, food, spell, side-facing

        vec = torch.tensor([
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0],  
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0],
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0],
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0],
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0],
            [1,0,0,0,0], 
            [0,1,0,0,0], 
            [0,0,1,0,0], 
            [0,0,0,1,0], 
            [0,0,0,0,1],  
            [0,0,0,0,0]])
        
        return len(vec), vec
    
    @staticmethod
    def plot_grid(x, n_sample, n_rows, save_dir, w):
        # x: (n_sample, c, h, w)
        # curiosamente, nrow é o número de colunas ou número de items em uma linha

        ncols = n_sample//n_rows
        grid = make_grid(DiffusionUtils.norm_torch(x), nrow=ncols)

        save_image(grid, save_dir + f"run_image_w{w}.png")
        print('saved image at ' + save_dir + f"run_image_w{w}.png")
        
        return grid

    @staticmethod
    def get_default_transform():

        transform = transforms.Compose([
            transforms.ToTensor(), # from [0,255] to range [0.0, 1.0]
            transforms.Normalize((0.5,), (0.5,)) # range [-1, 1]
        ])

        return transform
    
    @staticmethod
    def perturb_input(x, t, ab_t, noise):
        # Função necessária para criar uma perturbação na imagem com um específico nível de ruído
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
    
    @staticmethod
    def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn,  w, save = False):
        ncols = n_sample//nrows
        # change to Numpy image format (h,w,channels) vs (channels,h,w)
        # alterar para o formato numpy de imagem (h, w, channel) -> (channel, h, w)
        sx_gen_store = np.moveaxis(x_gen_store, 2, 4)
                                       
        # normalizar para o range [0, 1] necessário para np.imshow
        nsx_gen_store = DiffusionUtils.norm_all(sx_gen_store, 
                                                sx_gen_store.shape[0], n_sample)   
        
        # criar um gif de imagens evoluindo pelo tempo, baseado no x_gen_store
        fig, axs = plt.subplots(nrows=nrows, 
                                ncols=ncols, 
                                sharex=True, 
                                sharey=True,
                                figsize=(ncols,nrows))
        
        ani = FuncAnimation(fig, 
                            DiffusionUtils.animate_diff, 
                            fargs=[nsx_gen_store, nrows, ncols, axs],  
                            interval=200, 
                            blit=False, 
                            repeat=True, 
                            frames=nsx_gen_store.shape[0]) 
        plt.close()
        
        if save:
            ani.save(save_dir + f"{fn}.gif", dpi=100, writer=PillowWriter(fps=5))

            print('saved gif at ' + save_dir + f"{fn}.gif")

        return ani

    @staticmethod
    def animate_diff(i, store, nrows, ncols, axs):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')

        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i, (row*ncols) +col]))

        return plots
    
    @staticmethod
    def create_folder(path, name, use_date = False):
        """
        Método responsável por criar a pasta no diretório passado como parâmetro.
        """
        if use_date:
            dt = datetime.now()
            day = dt.strftime("%d")
            mes = dt.strftime("%m")
            hour = dt.strftime("%H")
            mm = dt.strftime("%M")
            dirname_base = f"{day}{mes}_{hour}{mm}_"
            directory = dirname_base + name
        else:
            directory = name

        parent_dir = path

        full_path = os.path.join(parent_dir, directory)

        if os.path.isdir(full_path):
            return full_path
        else:
            os.mkdir(full_path)
            return full_path