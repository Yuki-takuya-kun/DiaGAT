import os
import pickle
from typing import List, Dict
from collections import defaultdict

import numpy as np
import time
from scipy.linalg import block_diag
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from src.common import DataProcessor


class DiaDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        
        

class DiaDataloader(pl.LightningDataModule):
    """
    this class is going to process dataset automatically
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.file_path = self.cfg.json_path
        self.keys = [
                    'doc_id', 'input_ids', 'input_masks', 'input_segments',
                    'sentence_length', 'nsentence_ids', 'piece2words',
                    'new_triplets', 'reply_matrix', 'pairs', 'entities',
                    'entity_matrix', 'relation_matrix', 'polarity_lists',
                    'pos', 'dep_head', 'dep_label', 'dep_head_matrix',
                    'dep_label_matrix', 'speakers', 'speaker_matrix', 'replies',
                    'token_index', 'thread_length', 'cross_mat'
                ]
        self.dataprocessor = DataProcessor(self.cfg)
    
    def setup(self, stage):
        if self.cfg.lang == 'en':
            file_path = self.file_path + '_en'
        elif self.cfg.lang == 'zh':
            file_path = self.file_path + '_zh'
        else:
            raise ValueError("language is not 'en' or 'zh'")
        self.train_dataset = DiaDataset(os.path.join(file_path, 'train_processed.pickle'))
        self.valid_dataset = DiaDataset(os.path.join(file_path, 'valid_processed.pickle'))
        self.test_dataset = DiaDataset(os.path.join(file_path, 'test_processed.pickle'))
    
    def dia_collate(self, batch:List[Dict]):
        data = defaultdict(list)
        batch_size = len(batch)
        for elem in batch:
            for key in self.keys:
                data[key].append(elem[key])
       
        # padding function
        padding_token = lambda x, length: F.pad(x, pad=[0, length-x.shape[0]], mode='constant', value=0)
        padding_sentence = lambda x, length: F.pad(x, pad=[0,0,0,length-x.shape[0]], mode='constant', value=0)
        
        max_length = max(map(max,data['sentence_length']))
        max_sen_num = max(map(len, data['sentence_length']))
        # process input ids
        input_ids = data['input_ids']
        input_ids = [torch.stack([padding_token(torch.IntTensor(elem), max_length) for elem in elems]) for elems in input_ids]
        input_ids = torch.stack([padding_sentence(elem, max_sen_num) for elem in input_ids])
        
        # process input masks
        input_masks = data['input_masks']
        input_masks = [torch.stack([padding_token(torch.IntTensor(elem), max_length) for elem in elems]) for elems in input_masks]
        input_masks = torch.stack([padding_sentence(elem, max_sen_num) for elem in input_masks])
        
        padding_matrix = lambda x, length: F.pad(x, pad=[0,length-x.shape[1],0,length-x.shape[1]], mode='constant', value=0)
        
        matrix_len = max(map(sum, data['sentence_length']))
        
        #process pos mask
        pos_ids = data['pos']
        pos_ids = torch.stack([padding_token(torch.IntTensor(elem), matrix_len) for elem in pos_ids])
        
        # head matrix
        head_matrix = data['dep_head_matrix']
        head_matrix = ~torch.stack([padding_matrix(torch.Tensor(elem), matrix_len) for elem in head_matrix]).to(torch.bool)
        # (N, S, S)
        #print(torch.diag(head_matrix[0]))
        # label matrix
        tok_edges = []
        tok_weights = []
        tok_batid = []
        label_matrix = data['dep_label_matrix']
        sentence_lengths = data['sentence_length']
        batch_lengths = list(map(sum, sentence_lengths))
        for i in range(batch_size):
            label_mat = torch.Tensor(label_matrix[i]).to(torch.long)
            indices = label_mat.nonzero().t()
            weights = label_mat[indices[0], indices[1]]
            tok_edges.append(indices + sum(batch_lengths[:i]))
            tok_weights.append(weights)
            tok_batid += [i]*indices.size(1)
        
        
        tok_edges = torch.cat(tok_edges, dim=1)
        tok_weights = torch.cat(tok_weights)
        tok_batid = torch.Tensor(tok_batid).to(torch.long)
            
        
        # speaker mmatrix, speaker matrix is speaker masks
        speaker_matrix = data['speaker_matrix']
        
        # reply matrix, reply matrix is sen conj matrix
        reply_matrix = data['reply_matrix']
        reply_matrix = [self.dataprocessor.encode_replies_path(matrix) for matrix in reply_matrix]
        reply_matrix = [reply_matrix[i]+speaker_matrix[i] for i in range(len(batch))]
        reply_matrix = torch.stack([padding_matrix(torch.Tensor(matrix), max_sen_num) for matrix in reply_matrix]).to(dtype=torch.bool) # (N, S, S)
        
        
        sen_edges = []
        sen_batid = []
        for i in range(batch_size):
            for j in range(max_sen_num):
                for k in range(max_sen_num):
                    if reply_matrix[i][j][k] == 1:
                        sen_edges.append([sum(sentence_lengths[i][:j])+sum(batch_lengths[:i]), sum(sentence_lengths[i][:k])+sum(batch_lengths[:i])])
                        sen_batid.append(i)
        sen_edges = torch.Tensor(sen_edges).to(torch.long)
        sen_edges = sen_edges.transpose(0, 1)
        
        sen_batid = torch.Tensor(sen_batid).to(torch.long)
        #reply_matrix = ~torch.stack([padding_matrix(torch.Tensor(matrix), max_sen_num) for matrix in reply_matrix]).to(dtype=torch.bool)
        
        # entity_matrix
        entity_matrix = data['entity_matrix']
        entity_matrix = torch.stack([F.pad(torch.Tensor(matrix), pad=[0,matrix_len-matrix.shape[1]], mode='constant', value=0) \
            for matrix in entity_matrix])
        
        # relation_matrix
        relation_matrix = data['relation_matrix']
        relation_matrix = torch.stack([padding_matrix(torch.Tensor(matrix), matrix_len) for matrix in relation_matrix]) #(N,S,S)
    
        # encode sentence indices
        sentence_indices = torch.empty((len(batch), max_sen_num)) # (N, L, L)
        sentence_indices = torch.nn.init.constant_(sentence_indices, 0).int()
        sentence_indices[:, 0] = 1
        for i, sentence_length in enumerate(data['sentence_length']):
            for j, slen in enumerate(sentence_length[:-1]):
                sentence_indices[i, j+1] = slen + sentence_indices[i, j]
        
        triplets = []
        for elems in data['new_triplets']:
            sub_triplets = []
            for triplet in elems:
                triplet = triplet[:7]
                sub_triplets.append(torch.IntTensor(triplet))
            triplets.append(sub_triplets)
        
        replies = data['replies']
        cross_mats = data['cross_mat']
        #print(cross_mats.shape)
        # from tqdm import tqdm
        # tqdm.write(str(reply_matrix))
        # tqdm.write(str(head_matrix))
        
        return input_ids, input_masks, pos_ids, label_matrix, reply_matrix, head_matrix, sentence_indices,\
            entity_matrix, relation_matrix, data['pairs'], triplets, data['sentence_length'], cross_mats, \
            data['token_index'], data['thread_length'], data['nsentence_ids'], sen_edges, sen_batid, tok_edges, \
                tok_weights, tok_batid
    
    def get_cross_mat(self, replies):
        cross_mats = []
        for reply in replies:
            reply = [r + 1 for r in reply]
            sen_len = len(reply)
            cross_mat = torch.zeros((sen_len, sen_len), dtype=int)
            threads = [[0]]
            cur_thread = []
            for i in range(1, sen_len+1):
                if i == sen_len or reply[i] != reply[i-1] + 1:
                    threads.append(cur_thread)
                    cur_thread = [i]
                else:
                    cur_thread.append(i)
            for i, threadi in enumerate(threads):
                for ii in threadi:
                    for j, threadj in enumerate(threads):
                        for jj in threadj:
                            if i == j: # this indicate that two sentence belongs to one thread
                                cross_mat[ii, jj] = cross_mat[jj, ii] = abs(ii - jj)
                            elif i * j == 0: # this indicate that two sentence of one is head sen
                                cross_mat[ii, jj] = cross_mat[jj, ii] = ii + jj + 1
                            else:
                                cross_mat[ii, jj] = cross_mat[jj, ii] = ii + jj + 2
            cross_mats.append(cross_mat)
        return cross_mats
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.dia_collate)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.cfg.batch_size, collate_fn=self.dia_collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, collate_fn=self.dia_collate)
        
if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    with open('src/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = AttrDict(cfg)
    loader = DiaDataloader(cfg)
    loader.setup()
    train_loader = loader.train_dataloader()