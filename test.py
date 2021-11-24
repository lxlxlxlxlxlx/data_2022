#%%
import json
from tarfile import XGLTYPE
import torch
import glob
import pandas as pd
from tqdm import tqdm
import json
import csv
from functools import reduce
import operator
from math import log
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from sklearn.metrics import f1_score, accuracy_score
import gzip
from utils import args
from metrics4rec import evaluate_all

#%%

def load_candidate_items(args_):
    print('loading candidate items...')
    ui_cands = {}
    ui_gt = {}
    with gzip.open(args_.kg_test_candidates_file, 'rt') as f:
       for line in f:
           cells = line.split()
           uid = int(cells[0])
           item_ids = [int(i) for i in cells[1:]]
           ui_cands[uid] = item_ids
    #return ui_cands
    #data = np.load(args_.kg_test_candidates_file)['candidates']
    # ui_cands = {}
    # ui_gt = {}
    with open(args_.kg_test_triples_file, 'rt') as f:
        for line in f:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            if uid not in ui_gt:
                ui_gt[uid] = [iid]
            else:
                ui_gt[uid].append(iid)

    return ui_cands, ui_gt

# def main():
#%%
print('loading transE embedding...')
ckpt = torch.load(args.transE_embedding_file, map_location=torch.device('cpu'))
embeds = ckpt['ent_embeddings.weight'].cpu().numpy()
rels = ckpt['rel_embeddings.weight'].cpu().numpy()

print(type(embeds), embeds.shape)
print(type(rels), rels.shape)

ui_cands, gt = load_candidate_items(args)
print(len(ui_cands))
#%%
device = 'cpu'

def transE_eval(embeds, rels, ui_cands):
    ui_scores = {}
    cnt = 0
    for uid in ui_cands:
        iids = ui_cands[uid]
        u_emb = embeds[uid] + rels[0]  # user + purchase
        i_embs = embeds[iids]
        scores = np.expand_dims(u_emb, 0) - i_embs
        scores = LA.norm(scores, ord=1, axis=1) * (-1)  # larger is better
        ui_scores[uid] = dict(zip(iids, scores.tolist()))
        cnt += 1
        if cnt % 5000 == 0:
            print(cnt)
    return ui_scores

def causal_eval(opti_ui, opti_path_rep):
    vae_model = torch.load(args.vae_model_path, map_location=torch.device(device))
    vae_model.eval()
    model = torch.load(args.rec_model_path, map_location=torch.device(device))
    model.eval()
    re = []
    with torch.no_grad():
        rep_z = vae_model.get_z(torch.from_numpy(opti_path_rep).to(torch.device(device)))
    X = np.concatenate([embeds[opti_ui].reshape(-1,200), rep_z], axis=-1)

    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(opti_ui, scores)

def ui_eval(X):
    print('cal ui score...')
    model = torch.load(args.rec_ui_model_path, map_location=torch.device(device))
    #print(model.device())
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(ui, scores)

def uip_eval(opti_ui, opti_path_rep):
    X = np.concatenate([embeds[opti_ui].reshape(-1,200), opti_path_rep], axis=-1)
    model = torch.load(args.rec_uip_model_path, map_location=torch.device(device))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(opti_ui, scores)

def i_sbt_u_eval(opti_ui):
    X = embeds[opti_ui]
    X = X[:,100:] - X[:,:100]
    model = torch.load(args.rec_isu_model_path, map_location=torch.device(device))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(opti_ui, scores)

def i_sbt_up_eval(opti_ui):
    X = embeds[opti_ui]
    X = X[:,100:] - X[:,:100] - rels[0]
    model = torch.load(args.rec_isup_model_path, map_location=torch.device(device))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(opti_ui, scores)

def i_sbt_up_n1_eval(opti_ui, opti_path_rep):
    X = embeds[opti_ui]
    X = X[:,100:] - X[:,:100] - rels[0]
    X = LA.norm(X, ord=1, axis=1).reshape(-1,1)
    model = torch.load(args.rec_isupn1_model_path, map_location=torch.device(device))
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(X).to(torch.device(device)))
    return dict_scores(opti_ui, scores)


def p_eval():
    pass

def dict_scores(ui, scores):
    print('dicting score...')
    ui_scores = {}
    for i, u in enumerate(ui[::1000][:,0].flatten()):
        ui_scores[u] = dict(zip(ui[i*1000:(i+1)*1000,1].tolist(), scores[i*1000:(i+1)*1000].tolist()))
    return ui_scores

#%%
print('map embedding to ui...')
ui = np.zeros((len(ui_cands)*1000, 2), dtype=np.int)
for i, u in enumerate(ui_cands):
    ui[1000*i:1000*(i+1), 0] = u
    ui[1000*i:1000*(i+1), 1] = ui_cands[u]
ui_emb = embeds[ui].reshape(-1,200)

#args.inter_data_dir = f"./{args.save_dir}/{args.mode}/"
opti_paths = np.load(args.test_data_dir+'opti_paths.npy')
# opti_path_rep = np.load(args.test_data_dir+'opti_path_rep.npy')

nodes_emb = embeds[opti_paths[:,:,0]]
edges_emb = rels[opti_paths[:,1:,1]]
nodes_emb = np.sum(nodes_emb, axis=-2)
edges_emb = np.sum(edges_emb, axis=-2)
opti_path_rep = (nodes_emb+edges_emb)/7


# opti_paths_emb = np.concatenate([nodes_emb, edges_emb],axis=1)

# def represent_avg(v):
#     return np.average(v, axis=-2)

# opti_path_rep = represent_avg(opti_paths_emb)

#%%
print('map embedding to ui...')
ui = np.zeros((len(ui_cands)*1000, 2), dtype=np.int)
for i, u in enumerate(ui_cands):
    ui[1000*i:1000*(i+1), 0] = u
    ui[1000*i:1000*(i+1), 1] = ui_cands[u]
ui_emb = embeds[ui].reshape(-1,200)

#args.inter_data_dir = f"./{args.save_dir}/{args.mode}/"
opti_paths = np.load(args.test_data_dir+'opti_paths.npy')
# opti_path_rep = np.load(args.test_data_dir+'opti_path_rep.npy')

nodes_emb = embeds[opti_paths[:,:,0]]
edges_emb = rels[opti_paths[:,1:,1]]
nodes_emb = np.sum(nodes_emb, axis=-2)
edges_emb = np.sum(edges_emb, axis=-2)
opti_path_rep = (nodes_emb+edges_emb)/7

# opti_paths_emb = np.concatenate([nodes_emb, edges_emb],axis=1)

# def represent_avg(v):
#     return np.average(v, axis=-2)

# opti_path_rep = represent_avg(opti_paths_emb)

#%%
opti_ui = opti_paths[:,[0,3],0]
opti_ui_emb = embeds[opti_ui].reshape(-1,200)

#%%
causal_scores = causal_eval(opti_ui, opti_path_rep)
print('evaluating scores...')
evaluate_all(causal_scores, gt, 10)

#%%
# ui_scores = transE_eval(embeds, rels, ui_cands)
# print('evaluating scores...')
# evaluate_all(ui_scores, gt, 10)
#%%
causal_scores = causal_eval(opti_ui, opti_path_rep)
print('evaluating scores...')
evaluate_all(causal_scores, gt, 10)
#%%
uip_scores = uip_eval(opti_ui, opti_path_rep)
print('evaluating scores...')
evaluate_all(uip_scores, gt, 10)
#%%
#uip_emb
ui_scores = ui_eval(ui_emb)
print('evaluating scores...')
evaluate_all(ui_scores, gt, 10)