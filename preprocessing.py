#%%
from utils import args
import numpy as np
from collections import defaultdict
import torch
from threading import Thread
import pickle
from time import time
from  multiprocessing import Process,Pool
import os
import gzip
import itertools
import numpy_indexed as npi
from tqdm import tqdm
import multiprocessing
import random

args.mode = 'train'
args.k = multiprocessing.cpu_count()-1
args.transE_embedding_file = 'Cell_Phones_and_Accessories/openke/checkpoint/transe_cross_25_epoch100_lr1_wd1e-06_new.ckpt'
if args.mode == 'train':
    args.inter_data_dir = args.train_data_dir 
else:
    args.inter_data_dir = args.test_data_dir 
#%%
class KnowledgeGraph():
    def __init__(self, kg_file, purchase_file):
        self.kg_file = kg_file
        self.purchase_file = purchase_file
        self.G = defaultdict(list)
        self.uip = defaultdict(list) # user purchased item
        self.ui = defaultdict(list)
        self._load_emb()
        self._load_kg()

    def _load_emb(self):
        ckpt = torch.load(args.transE_embedding_file, map_location=torch.device('cpu'))
        self.embeds = ckpt['ent_embeddings.weight'].cpu().numpy()
        self.rels = ckpt['rel_embeddings.weight'].cpu().numpy()

    def _load_kg(self):
        with open(self.kg_file, 'r') as f:
            for line in f:
                k, v, r = map(int, line.rstrip().split('\t'))
                self.G[k].append((v,r))
                self.G[v].append((k,89-r)) # reverse edge
        with open(self.purchase_file, 'r') as f:
            for line in f:
                k, v, r = map(int, line.rstrip().split('\t'))
                self.uip[k].append((v,r))
                self.ui[k].append(v)
                self.G[v].append((k,89-r)) # reverse edge
        # remove repeated edges
        for k in self.G:
            self.G[k] = list(set(self.G[k]))
        for k in self.uip:
            self.uip[k] = list(set(self.uip[k]))
        for k in self.ui:
            self.ui[k] = list(set(self.ui[k]))

    def get_k_hop_path(self, u, k, mode, items=None):
        path = []
        if mode == 'train':
            to_visit = [self.G[u].copy()]
            items = set(items)
        else:
            assert items is not None
            to_visit = [self.G[u].copy()+self.uip[u].copy()]
            items = set(items)
        current_path = [(u, -1)]
        d = 1
        while True:
            if d == k:
                for u,i in to_visit[-1]:
                    if u in items:
                        path.append(current_path+[(u,i)])
                current_path.pop()
                to_visit.pop()
                d -= 1
            try:
                u, p = to_visit[-1].pop()
                # if u, p == 149744, 32:
                #     print(current_path)
            except IndexError:
                d-=1
                try:
                    to_visit.pop()
                except IndexError:
                    break
                current_path.pop()
                continue
            current_path.append((u, p))
            # to_visit.append(self.G[u].copy()+self.uip[u].copy())
            # to_visit.append(set(list(zip(*(self.G[u].copy()+self.uip[u].copy())))[0]) - set(list(zip(*current_path)))[0])
            to_visit.append([])
            for uu, pp in set(self.G[u].copy()+self.uip[u].copy()):
                if uu not in set(list(zip(*current_path))[0]):
                    to_visit[-1].append((uu, pp))

            d += 1
        return path
    
    def get_opti_k_hop_path(self, u, k, mode, items=None):

        if mode == 'train':
            neg_items = random.choices(range(61254,108858),k=5*len(self.ui[u]))
            items = self.ui[u] + neg_items
        path = self.get_k_hop_path(u, k, mode, items=items)
        if len(path) == 0:
            if mode == 'train':
                return [], 0
            else:
                return []
        re = np.array(path, dtype=int)
        
        # In[37]:
        nodes = re[:,:,0]
        paths = re[:,:,1]

        # In[39]:
        dist = self.embeds[nodes[:,1:]] - self.embeds[nodes[:,:-1]] - self.rels[paths[:,1:]] # a->b  a+r=b
        rev_dist = self.embeds[nodes[:,:-1]] - self.embeds[nodes[:,1:]] - self.rels[89-paths[:,1:]] # b->a  b+r=a
        dist = np.linalg.norm(dist,ord=1, axis=-1) ###L1
        rev_dist = np.linalg.norm(rev_dist,ord=1, axis=-1) ###L1
        # In[41]:
        #dist = np.sum(np.linalg.norm(dist,ord=1, axis=-1), axis=-1) ###L1
        comp = dist>rev_dist
        mdist = np.minimum(dist, rev_dist)
        mdist = np.sum(mdist, axis=-1)

        ui, idx = npi.group_by(re[:,[0,3],0]).argmin(mdist)
        # In[44]:
        re[:,1:,1][comp] = 89 - re[:,1:,1][comp]# 如果需要知道哪些是反着的，这里可以加一个负号
        opti_paths = re[idx]
        
        if mode =='train':
            nodes_i = opti_paths[:,3,0]
            label = np.isin(nodes_i, self.ui[u]).astype(np.int)
            return opti_paths, label
        return opti_paths
#if __name__ == '__main__':
#%%
print(f'start {args.mode} preprocessing!')
KG = KnowledgeGraph(args.kg_other_triples_file, args.kg_train_triples_file)
if args.mode == 'test':
    test_cand = []
    print('loading test candidate data.')
    with gzip.open(args.kg_test_candidates_file, 'rt') as f:
        for line in f:
            test_cand.append(list(map(int, line.rstrip().split('\t'))))
#%%
if args.mode == 'train': 
    def f(i,k):
        b = []
        label = []
        u = list(KG.ui.keys())[i::k]
        if i ==0:
            for uu in tqdm(u):
                a, l = KG.get_opti_k_hop_path(uu,3,'train')
                if len(a)>0:
                    b.append(a)
                    label.append(l)

        else:
            for uu in u:
                a, l = KG.get_opti_k_hop_path(uu,3,'train')
                if len(a)>0:
                    b.append(a)
                    label.append(l)

        return b, label
else:
    def f(i,k):
        b = []
        u = test_cand[i::k]
        if i ==0:
            for uu in tqdm(u):
                a = KG.get_opti_k_hop_path(uu[0],3,'test',uu[1:])
                if len(a)>0:
                    b.append(a)
        else:
            for uu in u:
                a = KG.get_opti_k_hop_path(uu[0],3,'test',uu[1:])
                if len(a)>0:
                    b.append(a)
        return b, []
#%%
k=args.k
t = []
result = []
label = []
start = time()
pool = Pool(k)
for i in range(k):
    t.append(pool.apply_async(func=f, args=(i,k)))

pool.close()
pool.join()
for tt in t:
    re, l = tt.get()
    result += re
    label += l
print(f'Find all optimal k-hop path time: {(time()-start)/60} minutes.')

opti_paths = np.vstack(result)
labels = np.concatenate(label)
# In[48]:
#args.inter_data_dir = '/home/jupyter/Cell/train/'
args.inter_data_dir = args.train_data_dir 
if not os.path.exists(args.inter_data_dir):
    os.makedirs(args.inter_data_dir)
np.save(args.inter_data_dir+'opti_paths.npy',opti_paths)
if args.mode == 'train':
    np.save(args.inter_data_dir+'label.npy',labels)
print(f'optimal path saved. shape: {opti_paths.shape}' )
# In[32]:
nodes_emb = KG.embeds[opti_paths[:,:,0]]
edges_emb = KG.rels[opti_paths[:,1:,1]]
opti_paths_emb = np.concatenate([nodes_emb, edges_emb],axis=1)
# In[51]:
def represent_avg(v):
    return np.average(v, axis=-2)

rep = represent_avg(opti_paths_emb)
# In[53]:

np.save(args.inter_data_dir+'opti_path_rep.npy',rep)
print(f'optimal path representation saved. shape: {rep.shape}' )
# %%
