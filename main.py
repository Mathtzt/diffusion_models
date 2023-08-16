import torch
from torch.utils.data import DataLoader

from classes.diffusion_model import DiffusionModel
from classes.custom_dataset import CustomDataset
from utils.diffusion_utils import DiffusionUtils as du

### Treinamento ou avaliação ###
train_or_eval = "eval" # Opções: "train" | "eval"
# If eval
model_path = "./results/models/1608_1659_ddpm_models" 
model_to_eval_name = "model_16"

### Fluxo ###
if train_or_eval == "train":
    ### Inicializando modelo de difusão ddpm ###
    dm_model = DiffusionModel()
    dm_model.init_model()
    ### Carregando transform (pre-processamento) padrão
    transform = du.get_default_transform()
    ### Carregando dataset ###
    dataset = CustomDataset(sfilename = "./data/sprites_1788_16x16.npy", 
                            lfilename = "./data/sprite_labels_nc_1788_16x16.npy", 
                            transform = transform, 
                            null_context = False) # Sem contexto, por enquanto
    ### Carregando data loader ###
    dataloader = DataLoader(dataset, 
                            batch_size = dm_model.batch_size, 
                            shuffle = True, 
                            num_workers = 1)
    ### Carregando otimizador ###
    optim = torch.optim.Adam(dm_model.model.parameters(), lr = dm_model.lrate)
    ### Treinamento
    dm_model.train(dataloader = dataloader,
                optimizer = optim)

elif train_or_eval == "eval":
    ### Inicializando modelo de difusão ddpm ###
    dm_model = DiffusionModel()
    dm_model.init_model()

    dm_model.load(model_path = model_path,
                  model_name = model_to_eval_name)

    dm_model.visualize_samples(filename = "an_dmodel_16ep")