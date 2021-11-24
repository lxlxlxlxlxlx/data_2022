#%%
from utils import args
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from model import VAE
import os
args.mode='train'
#%%
class DataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

path_rep = np.load(args.inter_data_dir+'opti_path_rep.npy')
dataset = DataSet(path_rep)
#%%
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(path_rep)*0.8), len(path_rep)-int(len(path_rep)*0.8)])
#%%
batch_size = 512
dataloader = {'train':torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4),
              'val':torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle=True, num_workers=4)}
# %%
device = args.device
learning_rate = 0.001
z_dim = 10
vae_model = VAE(z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate, weight_decay=0)
# %%
num_epochs = 5
for epoch in range(num_epochs):
    for i, path in enumerate(dataloader['train']):
        vae_model.train()
        path = path.to(device)
        x_reconst, mu, log_var = vae_model(path)
        loss = vae_model.loss_function(x_reconst, path, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            vae_model.eval()
            val_loss = 0
            with torch.no_grad():
                for path in dataloader['val']:
                    path = path.to(device)
                    x_reconst, mu, log_var = vae_model(path)
                    loss = vae_model.loss_function(x_reconst, path, mu, log_var)
                    val_loss += loss.item()
            
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Val_loss: {:.4f},' 
                  .format(epoch+1, num_epochs, i+1, len(dataloader), loss.item(), val_loss/len(dataloader['val'])))
# %%
if not os.path.exists('Cell/model'):
    os.mkdir('Cell/model')
torch.save(vae_model, args.vae_model_path)
# %%
