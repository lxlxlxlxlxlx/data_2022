#%%
from utils import args
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from metrics4rec import evaluate_all
from model import Rec_model
from numpy import linalg as LA
# %%
def train_rec(model, dataloader, lr, device, wd=1e-5,  num_epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    rec_loss = nn.BCELoss()
    for epoch in range(num_epochs):
        for i, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            score = model(data)
            loss = rec_loss(score, label) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
# %%
def load_data():
    ckpt = torch.load(args.transE_embedding_file, map_location=torch.device('cpu'))
    embeds = ckpt['ent_embeddings.weight'].cpu().numpy()
    rels = ckpt['rel_embeddings.weight'].cpu().numpy()
    #KG = KnowledgeGraph(args.kg_other_triples_file, args.kg_train_triples_file)

    opti_paths = np.load(args.train_data_dir+'opti_paths.npy')
    ui = opti_paths[:,[0,-1],0]
    label = np.load(args.train_data_dir+'label.npy')
    # item: 61254-108857 gen reverse label
    # rand = np.random.randint(61254, 108858, size=ui.shape[0])
    # label = np.random.randint(0, 2, size=ui.shape[0])
    # ui[:,1] = rand*(1-label) + ui[:,1]*label

    nodes_emb = embeds[opti_paths[:,:,0]]
    edges_emb = rels[opti_paths[:,1:,1]]
    opti_paths_emb = np.concatenate([nodes_emb, edges_emb],axis=1)
    def represent_avg(v):
        return np.average(v, axis=-2)
    path_rep = represent_avg(opti_paths_emb)


    ui_emb = embeds[ui].reshape(-1,200)

    vae_model = torch.load(args.vae_model_path).to(args.device)
    vae_model.eval()

    # path_rep = np.load(args.train_data_dir+'opti_path_rep.npy')

    with torch.no_grad():
        path_z = vae_model.get_z(torch.from_numpy(path_rep).to(args.device))
    path_z = path_z.detach().cpu().numpy()
    uiz_emb = np.concatenate((ui_emb, path_z), axis=-1)
    
    uip_emb = np.concatenate((ui_emb, path_rep), axis=-1)

    i_sbt_u = ui_emb[:,100:] - ui_emb[:,:100]
    i_sbt_up = ui_emb[:,100:] - ui_emb[:,:100] - rels[0]

    i_sbt_up_n1 = LA.norm(i_sbt_up, ord=1, axis=1).reshape(-1,1)

    return uiz_emb, ui_emb, uip_emb, i_sbt_u, i_sbt_up, i_sbt_up_n1, label
#%%
class Rec_DataSet(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label.reshape(-1,1).astype(np.float32)
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)
# train_selector = (np.random.rand(rec_dataset.__len__()) < 0.8).astype(int)
# train_idx, test_idx = torch.utils.data.random_split(np.arange(rec_input_data.shape[0]), [int(rec_input_data.shape[0]*0.8), int(rec_input_data.shape[0])-int(rec_input_data.shape[0]*0.8)])

# %%
def train_rec_model(input_size, train_data, label, args, model_save_path, learning_rate = 0.001,  num_epochs=2):
    train_dataset = Rec_DataSet(train_data, label)
    rec_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    rec_model = Rec_model(input_size=input_size, h_dim=[128,64,32]).to(args.device)
    train_rec(rec_model, rec_dataloader, learning_rate, args.device, num_epochs=num_epochs)
    torch.save(rec_model, model_save_path)
# rec_model = torch.load('inter_data/rec_model.pth')
#%%
if __name__ == '__main__':
    uiz_emb, ui_emb, uip_emb, i_sbt_u, i_sbt_up, i_sbt_up_n1, label = load_data()
    train_rec_model(210, uiz_emb, label, args, args.rec_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(200, ui_emb, label, args, args.rec_ui_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(300, uip_emb, label, args, args.rec_uip_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(100, i_sbt_u, label, args, args.rec_isu_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(100, i_sbt_up, label, args, args.rec_isup_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(1, i_sbt_up_n1, label, args, args.rec_isupn1_model_path, learning_rate=0.001, num_epochs=5)
    train_rec_model(100, np.abs(i_sbt_up), label, args, args.rec_isup_abs_model_path, learning_rate=0.001, num_epochs=5)
# %%
'''

test_ui = ui[test_idx]
# makeup_i = np.array([np.random.choice(list(range(61254, test_ui[i,1]))+list(range(test_ui[i,1], 108858)), size=(1,100), replace=False) for i in range(test_ui.shape[0])])
makeup_i = np.random.choice(list(range(61254, 108858)), size=(test_ui.shape[0], 100), replace=True)
makeup_i = makeup_i.reshape(-1,1)
test_u = np.repeat(test_ui[:,0], 100, axis=0).reshape(-1,1)
makeup_ui = np.concatenate((test_u, makeup_i), axis=-1)
#%%
makeup_ui_emb = embeds[makeup_ui].reshape(-1,200)
test_path_z = path_z[test_idx]
test_path_z_r = np.repeat(test_path_z, 100, axis=0)
makeup_ui_emb = np.concatenate((makeup_ui_emb, test_path_z_r), axis=-1) #210

test_ui_emb = embeds[test_ui].reshape(-1,200)
test_ui_emb = np.concatenate((test_ui_emb, test_path_z), axis=-1) #210

makeup_ui_emb = makeup_ui_emb.reshape(-1,100,210)
test_rec_input_data = np.concatenate((makeup_ui_emb, test_ui_emb.reshape(-1,1,210)), axis=1)
#%%
rec_model.eval()
with torch.no_grad():
    scores = rec_model(torch.from_numpy(test_rec_input_data.reshape(-1,210)).to(device))
# %%
scores = scores.reshape(-1,101).detach().cpu().numpy()
# %%
gt = dict(test_ui)
for uu in gt:
    gt[uu] = [gt[uu]]
# %%
test_i = np.concatenate((makeup_i.reshape(-1,100), test_ui[:,1:2]), axis=-1)
# %%
tmp = [dict(zip(test_i[i],scores[i])) for i in range(test_i.shape[0])]
ui_score = dict(zip(test_ui[:,0], tmp))
# %%
eval_result = evaluate_all(ui_score, gt, 10) # higher is better
# %%
print(eval_result)
'''