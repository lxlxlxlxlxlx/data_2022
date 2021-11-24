from easydict import EasyDict as edict
import torch
import os

args = edict()
# arguments
args.mode = 'train'
args.dataset = "Cell_Phones_and_Accessories"
args.k = 8

# settings
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.save_dir = args.dataset.split('_')[0]

# will be use
args.vae_model_path = f"./{args.save_dir}/model/vae_model.pth"
args.rec_model_path = f"./{args.save_dir}/model/rec_model.pth"
args.rec_ui_model_path = f"./{args.save_dir}/model/rec_ui_model.pth"
args.rec_uip_model_path = f"./{args.save_dir}/model/rec_uip_model.pth"
args.rec_isu_model_path = f"./{args.save_dir}/model/rec_isu_model.pth"
args.rec_isup_model_path = f"./{args.save_dir}/model/rec_isup_model.pth"
args.rec_isupn1_model_path = f"./{args.save_dir}/model/rec_isupn1_model.pth"
args.rec_isup_abs_model_path = f"./{args.save_dir}/model/rec_isup_abs_model.pth"
args.train_data_dir = f"./{args.save_dir}/train/"
args.test_data_dir = f"./{args.save_dir}/test/"



# KG related file
args.tmp_dir = args.dataset
args.kg_users_file = '{}/kg_users_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_items_file = '{}/kg_items_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_entities_file = '{}/kg_entities_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_relations_file = '{}/kg_relations_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_triples_file = '{}/kg_train_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_gt_file = '{}/kg_train_gt_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_other_triples_file = '{}/kg_other_triples_{}.txt'.format(args.tmp_dir, args.dataset)

args.kg_test_triples_file = '{}/kg_test_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_test_candidates_file = '{}/kg_test_candidates_{}.txt.gz'.format(args.tmp_dir, args.dataset)

args.transE_embedding_file = '{}/openke/checkpoint/transe_cross_25_epoch100_lr1_wd1e-06_new.ckpt'.format(args.tmp_dir, args.dataset)

'''
from easydict import EasyDict as edict
import torch

args = edict()

args.tmp_dir = "Cell_Phones_and_Accessories"
args.dataset = "Cell_Phones_and_Accessories"

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args.vae_model_path = "./inter_data/vae_model.pth"
args.rec_model_path = 'inter_data/rec_model.pth'
args.rec_ui_model_path = 'inter_data/rec_ui_model.pth'
args.rec_uip_model_path = 'inter_data/rec_uip_model.pth'
# KG related files
args.kg_users_file = '{}/kg_users_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_items_file = '{}/kg_items_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_entities_file = '{}/kg_entities_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_relations_file = '{}/kg_relations_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_triples_file = '{}/kg_train_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_train_gt_file = '{}/kg_train_gt_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_other_triples_file = '{}/kg_other_triples_{}.txt'.format(args.tmp_dir, args.dataset)

args.kg_test_triples_file = '{}/kg_test_triples_{}.txt'.format(args.tmp_dir, args.dataset)
args.kg_test_candidates_file = '{}/kg_test_candidates_{}.txt.gz'.format(args.tmp_dir, args.dataset)

args.transE_embedding_file = '{}/openke/transe_epoch_100_lr_0.002.ckpt'.format(args.tmp_dir, args.dataset)
'''