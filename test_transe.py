#%%
import json
import multiprocessing
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
from sklearn.metrics import f1_score, accuracy_score
import gzip
import os
import sys
from multiprocessing import Pool

from config import args
from metrics4rec import evaluate_all


# data_dir = '/content/drive/MyDrive/www_2022/data_and_src/{}/openke/'.format(args.dataset)
# data_dir = '/home/g0610sep2021/Cell_Phones_and_Accessories/OpenKE/checkpoint/'
# model_name = 'transe_cross_5.ckpt'

def load_candidate_items(args_):
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
    with open(args.kg_test_triples_file, 'rt') as f:
        for line in f:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            if uid not in ui_gt:
                ui_gt[uid] = [iid]
            else:
                ui_gt[uid].append(iid)

    return ui_cands, ui_gt



def main(model_name):
    ckpt = torch.load(data_dir + model_name)
    embeds = ckpt['ent_embeddings.weight'].cpu().numpy()
    rels = ckpt['rel_embeddings.weight'].cpu().numpy()
    # print(type(embeds), embeds.shape)
    # print(type(rels), rels.shape)

    # print(embeds[0])
    # print(rels[0])



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
        # if cnt % 5000 == 0:
        #     print(cnt)

    #gt = {}
    #with open(args.kg_test_triples_file) as f:
    #    for line in f:
    #        cells = line.strip().split('\t')
    #        uid, iid = int(cells[0]), int(cells[1])
    #        if uid not in gt:
    #            gt[uid] = []
    #        gt[uid].append(iid)
    print()
    msg, res = evaluate_all(ui_scores, gt, 10, print_result=False)
    print(model_name+':\n', msg)
    sys.stdout.flush()

if __name__ == '__main__':
    ui_cands, gt = load_candidate_items(args)
    print(len(ui_cands))
    data_dir = '/home/g0610sep2021/Cell_Phones_and_Accessories/OpenKE/checkpoint/'

    k = multiprocessing.cpu_count()-1
    pool = Pool(k)
    t = []
    for model_name in os.listdir(data_dir): 
        if 'new' in model_name:
            print('aaa'+model_name)
            t.append(pool.apply_async(func=main, args=(model_name,)))
    pool.close()
    pool.join()