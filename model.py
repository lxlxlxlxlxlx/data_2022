#%%
#from preprocessing import *
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from metrics4rec import evaluate_all
#%%
class VAE(nn.Module):
    def __init__(self, input_size=100, h_dim=128, z_dim=10):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, h_dim)
        self.fc11 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc44 = nn.Linear(h_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_size)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        h = F.relu(self.fc11(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc4(z))
        h = F.relu(self.fc44(z))
        return F.sigmoid(self.fc5(h))
    
    def get_z(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    def loss_function(self, x_reconst, x, mu, log_var):
        MSE = F.mse_loss(x_reconst, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD


#%%
class Rec_model(nn.Module):
    def __init__(self, input_size=210, h_dim=[128]):
        super().__init__()
        fc_list = [nn.Linear(input_size, h_dim[0])]
        for i in range(len(h_dim)-1):
            fc_list.append(nn.ReLU())
            fc_list.append(nn.Linear(h_dim[i], h_dim[i+1]))
        fc_list.append(nn.ReLU())
        fc_list.append(nn.Linear(h_dim[-1], 1))
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x): # u i z . i(i+ i-)
        return torch.sigmoid(self.fc(x))

