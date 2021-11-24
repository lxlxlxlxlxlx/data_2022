from os import defpath
#from utils import args
import numpy as np
from collections import defaultdict
#import torch
from tqdm import tqdm

#%%
class KnowledgeGraph():
    def __init__(self, kg_file, purchase_file):
        self.kg_file = kg_file
        self.purchase_file = purchase_file
        self.G = defaultdict(list)
        self.uip = defaultdict(list) # user purchased item
        self.ui = defaultdict(list)
        self._load_kg()


    def _load_kg(self):
        with open(self.kg_file, 'r') as f:
            for line in f:
                k, v, r = map(int, line.rstrip().split('\t'))
                self.G[k].append((v,r))
                self.G[v].append((k,r)) # reverse edge
        with open(self.purchase_file, 'r') as f:
            for line in f:
                k, v, r = map(int, line.rstrip().split('\t'))
                self.uip[k].append((v,r))
                self.ui[k].append(v)
                self.G[v].append((k,r)) # reverse edge

        # remove repeated edges
        for k in self.G:
            self.G[k] = list(set(self.G[k]))
        for k in self.uip:
            self.uip[k] = list(set(self.uip[k]))
        for k in self.ui:
            self.ui[k] = list(set(self.ui[k]))

    def get_k_hop_path(self, u, k):
        path = []
        items = set(self.ui[u])
        to_visit = [self.G[u].copy()]
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


