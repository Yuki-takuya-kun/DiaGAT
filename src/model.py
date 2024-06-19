
import math
from typing import List, Dict, Optional
from itertools import accumulate
from tqdm import tqdm

import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torch_geometric.nn import GAT
import lightning.pytorch as pl

from transformers import AutoModel

from src.common import DataProcessor, WarmupCosineDecayLR
from src.metrics import DiaMetric

def aggregate(x:torch.Tensor, method:str='mean'):
    """
    This function is going to aggregate tensors in different way such as mean, max pooling
    Parameters:
        x: the Tensor that needs to aggregate, its shape must be (N, *)
        method: aggregate way. 'mean', 'max', 'min', 'sum' is support
    """
    if method == 'mean':
        return torch.mean(x, dim=0)
    elif method == 'max':
        return torch.max(x, dim=0)[0]
    elif method == 'min':
        return torch.min(x, dim=0)[0]
    elif method == 'sum':
        return torch.sum(x, dim=0)
    else:
        raise ValueError(f'aggregate method:{method} is not support')


class MLP(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 output_size:int,
                 hidden_layer_num:int = 0,
                 dropout:float = 0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.hidden_layer_num = hidden_layer_num
        
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.hidden_layer_num)])
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout)
        self.layernorm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.LeakyReLU(inplace=False)
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        x_ = self.input_layer(x)
        x_ = self.activation(x_)
        x_ = self.dropout(x_)
        for hidden_layer in self.hidden_layers:
            x_ = hidden_layer(x_)
            x_ = self.activation(x_)
            x_ = self.dropout(x_)
        x_ = self.output_layer(x_)
        if self.input_size == self.output_size:
            x = self.layernorm(x + x_)
        else: x = x_
        return x

class PLM(nn.Module):
    """
    PLM is used to embed tokens into vectors
    """
    def __init__(self, 
                 plm_name:str,
                 plm_abla:bool=False,
                 cache_dir:str=None):
        super().__init__()
        self.plm_abla = plm_abla
        self.plm = AutoModel.from_pretrained(plm_name, cache_dir=cache_dir)
        self.config = self.plm.config
        
        self.hidden_size = self.config.hidden_size
        
    def forward(self, input_ids:torch.Tensor, input_masks:torch.Tensor):
        if input_ids.shape != input_masks.shape:
            raise ValueError(f'shape of input_ids and input_masks must be same, but get {input_ids.shape} and {input_masks.shape}')
        if self.plm_abla:
            return self.plm(input_ids=input_ids)
        else:
            return self.plm(input_ids=input_ids, attention_mask=input_masks).last_hidden_state

class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_size:int, theta:float = 10000.0):
        super(RotaryEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.theta = theta
        self.freqs = 1.0 / (self.theta**(torch.arange(0, self.hidden_size, 2)[:(self.hidden_size// 2)].float()/self.hidden_size))
        self.max_seq_len = 0
        
    def _get_freqs_cis(self, x, seq_len, indices=None):
        if indices is not None:
            t = indices.cpu()
            freqs = torch.outer(t, self.freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(x.device)
            return freqs_cis
        if seq_len > self.max_seq_len:
            t = torch.arange(seq_len)
            freqs = torch.outer(t, self.freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(x.device)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        return self.freqs_cis[:seq_len]
    
    def forward(self, x:torch.Tensor, indices=None):
        # x: (N, S, S)
        seq_len = x.size(1)
        freqs_cis = self._get_freqs_cis(x, seq_len, indices)
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        x = torch.view_as_complex(x)
        if len(x.shape) != len(freqs_cis.shape):
            freqs_cis = freqs_cis.unsqueeze(1)
        x = torch.view_as_real(x * freqs_cis).flatten(2).to(dtype=x.dtype)
        return x

class MultiHeadGraphAttentionNetwork(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 num_heads:int = 8,
                 dropout:float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = self.hidden_size // self.num_heads
        self.w = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Linear(2*self.head_dim, 1) #(2*HD, 1)
        self.a_w = nn.Linear(3*self.head_dim, 1) #(3*HD, 1)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x:torch.Tensor, edges:torch.Tensor, weight:Optional[torch.Tensor]=None):
        # x: (S, D), edges:(2,E), weight: (E, D)
        #print('0', x)
        residule = x
        seq_len = x.size(0)
        edge_size = edges.size(1)
        x1 = x[edges[0]] #(E, D)
        x2 = x[edges[1]] #(E, D)
        out = self.v_proj(x) #(S, D)
        out = out.view(seq_len, self.num_heads, -1).transpose(0, 1) #(H, S, HD)
        x = torch.cat([x1, x2], dim=1) #(E, 2*H)
        x = self.w(x) #(E, 2*H)
        #print('1', x)
        if weight is not None:
            x = torch.cat([x, weight], dim=1) #(E, 3*H)
            x = x.view(edge_size, self.num_heads, -1).transpose(0, 1) #(H, E, 3*HD)
            x = F.leaky_relu(x)
            w = self.a_w(x) #(H, E, 1)
        else:
            x = x.view(edge_size, self.num_heads, -1).transpose(0, 1) #(H, E, 2*HD)
            x = F.leaky_relu(x)
            w = self.a(x) #(H, E, 1)
        w = w.squeeze(-1) #(H, E)
        #w = F.leaky_relu(w) #(H, E)
        weight_mat = w.new_zeros((self.num_heads, seq_len, seq_len)) #(H, S, S)
        weight_mat[:] = float('-inf')
        weight_mat[:, edges[0], edges[1]] = w #(H, S, S)
        weight_mat = weight_mat.softmax(dim=-1) #(H, S, S)
        weight_mat = torch.nan_to_num(weight_mat)
        x = torch.matmul(weight_mat, out).transpose(0, 1) #(S, H, HD)
        x = F.elu(x) #(S, H, HD)
        x = x.contiguous().view(seq_len, self.hidden_size) #(S, D)
        x += residule
        x = self.layer_norm(x)
        x = self.dropout(x)
        #print('3', x)
        return x
        
        

class SGAT(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 num_heads:int = 8,
                 dropout:float = 0.0):
        super(SGAT, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.gat_layer = MultiHeadGraphAttentionNetwork(self.hidden_size, self.num_heads, self.dropout)
        #self.attn_layers = MultiHeadGraphAttentionNetwork(self.hidden_size, self.num_heads)
        
    def forward(self, x:torch.Tensor, edges:torch.Tensor):
        # x:(S, H), edges:(2, E)
        return self.gat_layer(x, edges)


class TGAT(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 aggregate_method:str = 'mean'):
        super(TGAT, self).__init__()
        self.hidden_size = hidden_size
        self.scale_factor = 1 / math.sqrt(hidden_size)
        self.dropout = dropout
        self.num_heads = num_heads
        self.aggregate_method = aggregate_method
        
        self.dropout_layer = nn.Dropout(self.dropout, inplace=False)
        self.gat_layer = MultiHeadGraphAttentionNetwork(self.hidden_size, self.num_heads, self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x:torch.Tensor, edges:torch.Tensor, weights:torch.Tensor):
        # x: (N, S, H), dep_mat: (N, S, S, E/H), conj_mat: (N, S, S)
        
        return self.gat_layer(x, edges, weights)

class MyGAT(nn.Module):
    def __init__(self, hidden_size:int,
                 num_heads:int,
                 gat_layer_num:int,
                 dropout:float,
                 mode='pos'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gat_layer_num = gat_layer_num
        self.dropout = dropout
        self.mode = mode
        if self.mode != 'pos':
            self.sgat_layers = nn.ModuleList([
                SGAT(self.hidden_size, self.num_heads, self.dropout)
                for _ in range(self.gat_layer_num)])
        self.tgat_layers = nn.ModuleList([
            TGAT(self.hidden_size, self.num_heads, self.dropout)
            for _ in range(self.gat_layer_num)])
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        
    def forward(self, x, sen_edges, tok_edges, tok_edge_emb):
        for i in range(self.gat_layer_num):
            residue = x
            if self.mode != 'pos':
                x = self.sgat_layers[i](x, sen_edges) # (N, L, E)
            x = self.tgat_layers[i](x, tok_edges, tok_edge_emb) # (N, S, E)
            x += residue
            #    self.layer_norm(x)
        return x

    def extract_submat_indices(self, x:torch.Tensor, indices:torch.Tensor):
        # x:(N, S, E), indices:(N, L)
        bs = x.size(0)
        x = torch.concat([torch.zeros((bs, 1, self.hidden_size), device=x.device), x], dim=1) # (N, S+1, E)
        submat = torch.stack([x[i].index_select(0, indices[i]) for i in range(bs)]) # (N, L, E)
        return submat
    
    def assign_submat_indices(self, x:torch.Tensor, submat:torch.Tensor, indices:torch.Tensor):
        # x:(N, S, E), submat:(N, L, E), indices:(N, L) 
        for i in range(x.size(0)):
            for j, idx in enumerate(indices[i]):
                if idx == 0: continue
                x[i, idx-1] = submat[i, j]
        x = x.contiguous()
        return x

class EdgeLogiter(nn.Module):
    def __init__(self, hidden_size, edge_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.edge_classes = edge_classes
        self.scale_factor = 1 / math.sqrt(hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size*self.edge_classes)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size*self.edge_classes)
        self.rotary_embedding = RotaryEmbedding(hidden_size=self.hidden_size, theta=10000)
    
    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """
        
        seq_len, num_classes = qw.shape[:2]
        token_index = torch.Tensor(token_index).to(qw.device)
        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len,num_classes ])
        #logits[..., 0] = float('inf')
        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i+1]
                cstart, cend = accu_index[j], accu_index[j+1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]
                x, y = token_index[rstart:rend], token_index[cstart:cend]
                #print(rstart, rend, cstart, cend)
                # This is used to compute relative distance, see the matrix in Fig.8 of our paper
                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                cur_qw = self.rotary_embedding(cur_qw, x)
                cur_kw = self.rotary_embedding(cur_kw, y)

                # matrix multiplication
                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()
                #print(pred_logits)
                logits[rstart:rend, cstart:cend] = pred_logits
        logits = logits.permute(2, 0, 1)
        
        return logits 
    
    def get_ro_embedding(self, qw, kw, token_index, thread_lengths):
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i])
            logits.append(pred_logits)
        logits = torch.stack(logits) 
        return logits 
    
    def forward(self, x:torch.Tensor, token_index, thread_length):
        # x: (N, S, E)
        batch_size, seq_len = x.size(0), x.size(1)
        q = self.q_proj(x) # (N, S, E*C)
        q = q.view(batch_size, seq_len, self.edge_classes, self.hidden_size) # (N, S, C, E)
        #q = q.transpose(1,2).contiguous() # (N, C, S, E)
        k = self.k_proj(x) # (N, S, E*C)
        k = k.view(batch_size, seq_len, self.edge_classes, self.hidden_size) # (N, S, C, E)
        #k = k.transpose(1,2).contiguous() # (N, C, S, E)
        
        w = self.get_ro_embedding(q, k, token_index, thread_length)
        # w = torch.matmul(q, k.transpose(2,3)) # (N, C, S, S)
        return w
           
class DiaGAT(pl.LightningModule):
    def __init__(self,
                 plm_name:str,
                 pos_nums:int,
                 dep_nums:int,
                 token_label_num:int,
                 edge_classes:int,
                 dataprocessor:DataProcessor,
                 schedulers_cfg:Dict,
                 gat_layer_num:int = 1,
                 node_gat_layer_num:int = 1,
                 edge_gat_layer_num:int = 1,
                 sen_gat_layer_num:int = 1,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 node_loss_weight: float = 1.0,
                 edge_loss_weight: float = 1.0,
                 polar_loss_weight: float = 1.0,
                 aggregate_method:str = 'mean',
                 huggingface_dir:str = None,
                 freeze_modules:Dict = {},
                 **kwargs):
        super().__init__()
        self.hidden_size = 0
        self.plm_name = plm_name
        self.dataprocessor = dataprocessor
        self.schedulers_cfg = schedulers_cfg
        self.token_label_num = token_label_num
        self.edge_classes = edge_classes
        self.gat_layer_num = gat_layer_num
        self.node_gat_layer_num = node_gat_layer_num
        self.edge_gat_layer_num = edge_gat_layer_num
        self.sen_gat_layer_num = sen_gat_layer_num
        self.num_heads = num_heads
        self.dropout = dropout
        self.node_loss_weight = node_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.polar_loss_weight = polar_loss_weight
        self.aggregate_method = aggregate_method
        self.freeze_modules = freeze_modules
        
        self.plm = PLM(self.plm_name, plm_abla=kwargs['plm_abla'], cache_dir=huggingface_dir)
        self.hidden_size = self.plm.hidden_size
        
        self.pos_embedding = nn.Embedding(pos_nums, self.hidden_size, padding_idx=0)
        self.dep_embedding = nn.Embedding(dep_nums, self.hidden_size, padding_idx=0)
        
        self.node_gat = MyGAT(hidden_size=self.hidden_size, num_heads=self.num_heads, gat_layer_num=self.gat_layer_num,
                            dropout=self.dropout, mode='pos')
        self.edge_gat = MyGAT(hidden_size=self.hidden_size, num_heads=self.num_heads, gat_layer_num=self.gat_layer_num,
                            dropout=self.dropout, mode='rel')
        self.sen_gat = MyGAT(hidden_size=self.hidden_size, num_heads=self.num_heads, gat_layer_num=self.gat_layer_num,
                            dropout=self.dropout, mode='pos')
        
        self.node_classifier = MLP(self.hidden_size, self.hidden_size, self.token_label_num)
        self.edge_regressioner = MLP(self.hidden_size, self.hidden_size, self.edge_classes)
        self.polar_classifier = MLP(3*self.hidden_size, self.hidden_size, 3)
        
        self.edge_logiter = EdgeLogiter(self.hidden_size, self.edge_classes)
        
        self.train_metric = DiaMetric(dataprocessor=self.dataprocessor)
        self.valid_metric = DiaMetric(dataprocessor=self.dataprocessor)
        self.test_metric = DiaMetric(dataprocessor=self.dataprocessor)
        
        self.pos_modules = [self.pos_embedding, self.dep_embedding, self.node_gat, self.edge_gat,
                            self.node_classifier, self.edge_regressioner, self.polar_classifier,
                            self.edge_logiter]

    def training_step(self, batch:List, batch_idx):
        # batch: (input_ids:(N, S), input_masks:(N, S), dep_mat:(N, S, S), sen_conj_mat:(N, L, L),
        # token_conj_mat:(N, S, S), cls_indices:(N, L), node_logits_label:(N, S, 7),
        # edge_logits_label:(N, S, S), triplets:List, sentence_lengths:List)
        input_ids, input_masks, pos_ids, dep_mat, sen_conj_mat, token_conj_mat, cls_indices, \
            node_logits_label, edge_logits_label, pairs, triplets, sentence_lengths, cross_mats, \
            token_index, thread_length, sentence_ids, sen_edges, sen_batid, tok_edges, \
                tok_weights, tok_batid = batch
        node_logits, edge_logits, triplet_label_logits, outputs = self(input_ids, input_masks, pos_ids, dep_mat,
                                                              sen_conj_mat, token_conj_mat, cls_indices,
                                                              triplets, sentence_lengths, token_index, thread_length,
                                                              sen_edges, sen_batid, tok_edges, tok_weights, tok_batid)
        weights = self.node_loss_weight + self.edge_loss_weight + self.polar_loss_weight
        node_loss = self.cal_node_loss(node_logits, node_logits_label)
        self.log("train_node_loss", node_loss, prog_bar=True)
        
        # edge_mask = outputs['edge_mask']
        # edge_logits_label = edge_logits_label.permute(0,2,3,1) #(N, S, S, 2)
        # edge_logits_label = edge_logits_label*edge_mask
        # edge_logits_label = edge_logits_label.permute(0,3,1,2) #(N, 2, S, S)
        #print(edge_logits_label.shape)
        edge_loss = self.cal_edge_loss(edge_logits, edge_logits_label)
       
        self.log("train_edge_loss", edge_loss, prog_bar=True)
        
        polar_loss = self.cal_polar_loss(triplet_label_logits, triplets)
        self.log("train_polar_loss", polar_loss, prog_bar=True)
        
        self.train_metric.update(outputs, node_logits_label, edge_logits_label,
                                 pairs, triplets, sentence_lengths, cross_mats,
                                 sentence_ids)
        
        # return self.node_loss_weight*node_loss + self.edge_loss_weight*edge_loss + \
        #     self.polar_loss_weight*polar_loss
        return  node_loss + edge_loss + polar_loss
    
    def validation_step(self, batch:List, batch_idx):
        input_ids, input_masks, pos_ids, dep_mat, sen_conj_mat, token_conj_mat, cls_indices, \
            node_logits_label, edge_logits_label, pairs, triplets, sentence_lengths, cross_mats, \
            token_index, thread_length, sentence_ids, sen_edges, sen_batid, tok_edges, \
                tok_weights, tok_batid = batch
        node_logits, edge_logits, triplet_label_logits, outputs = self(input_ids, input_masks, pos_ids, dep_mat,
                                                              sen_conj_mat, token_conj_mat, cls_indices,
                                                              triplets, sentence_lengths, token_index, thread_length,
                                                              sen_edges, sen_batid, tok_edges, tok_weights, tok_batid)
        
        node_loss = self.cal_node_loss(node_logits, node_logits_label)
        self.log("valid_node_loss", node_loss)
        # edge_mask = outputs['edge_mask']
        # edge_logits_label = edge_logits_label.permute(0,2,3,1) #(N, S, S, 2)
        # edge_logits_label = edge_logits_label*edge_mask
        # edge_logits_label = edge_logits_label.permute(0,3,1,2) #(N, 2, S, S)
        edge_loss = self.cal_edge_loss(edge_logits, edge_logits_label)
        self.log("valid_edge_loss", edge_loss)
        
        
        polar_loss = self.cal_polar_loss(triplet_label_logits, triplets)
        self.log("valid_polar_loss", polar_loss)
        
        self.valid_metric.update(outputs, node_logits_label, edge_logits_label,
                                 pairs, triplets, sentence_lengths, cross_mats,
                                 sentence_ids)
        
    def test_step(self, batch:List, batch_idx):
        input_ids, input_masks, pos_ids, dep_mat, sen_conj_mat, token_conj_mat, cls_indices, \
            node_logits_label, edge_logits_label, pairs, triplets, sentence_lengths, cross_mats, \
            token_index, thread_length, sentence_ids, sen_edges, sen_batid, tok_edges, \
                tok_weights, tok_batid = batch
        node_logits, edge_logits, triplet_label_logits, outputs = self(input_ids, input_masks, pos_ids, dep_mat,
                                                              sen_conj_mat, token_conj_mat, cls_indices,
                                                              triplets, sentence_lengths, token_index, thread_length,
                                                              sen_edges, sen_batid, tok_edges, tok_weights, tok_batid)
        
        node_loss = self.cal_node_loss(node_logits, node_logits_label)
        self.log("test_node_loss", node_loss)
        # edge_mask = outputs['edge_mask']
        # edge_logits_label = edge_logits_label.permute(0,2,3,1) #(N, S, S, 2)
        # edge_logits_label = edge_logits_label*edge_mask
        # edge_logits_label = edge_logits_label.permute(0,3,1,2) #(N, 2, S, S)
        edge_loss = self.cal_edge_loss(edge_logits, edge_logits_label)
        self.log("test_edge_loss", edge_loss)
        
        # polar_loss = self.cal_polar_loss(triplet_logits, triplets)
        # self.log("test_polar_loss", polar_loss)
        
        self.test_metric.update(outputs, node_logits_label, edge_logits_label,
                                pairs, triplets, sentence_lengths, cross_mats,
                                sentence_ids)
        
    def configure_optimizers(self):
        optimizer1 = optim.AdamW(self.plm.parameters())
        scheduler1 = WarmupCosineDecayLR(optimizer1, 
                                        self.schedulers_cfg['warmupCosineDecay']['warmup_epochs'],
                                        self.schedulers_cfg['total_epochs'],
                                        self.schedulers_cfg['warmupCosineDecay']['start_lr'],
                                        self.schedulers_cfg['warmupCosineDecay']['intermediate_lr'],
                                        self.schedulers_cfg['warmupCosineDecay']['end_lr'])
        # opt2_params = list(self.pos_modules[0].parameters())
        # for i in range(1, len(self.pos_modules)):
        #     opt2_params += list(self.pos_modules[i].parameters())
        # optimizer2 = optim.AdamW(opt2_params, lr=1e-3)
        # return ({
        #     'optimizer':optimizer1,
        # }, {'optimizer': optimizer2})
        return {'optimizer': optimizer1, 'lr_scheduler':scheduler1}
    
    def forward(self, input_ids:torch.Tensor, input_masks:torch.Tensor,
                pos_ids:torch.Tensor,
                dep_mat:torch.Tensor, sen_conj_mat:torch.Tensor,
                token_conj_mat:torch.Tensor, cls_indices:torch.Tensor,
                triplets:List[torch.Tensor], sentence_lengths:List, 
                token_index, thread_length, sen_edges, sen_batid, tok_edges,
                tok_weights, tok_batid):
        # input_ids:(N, S, T), input_masks:(N, S, T), dep_mat:(N, S, S), sen_conj_mat:(N, L, L), token_conj_mat:(N, S, S)
        # sen_batid #(S, 1)

        pos_emb = self.pos_embedding(pos_ids) # (N, S, E)
        #print(tok_weights.shape)
        dep_mat = self.dep_embedding(tok_weights)
        batch_size, seq_len, token_len = input_ids.size(0), input_ids.size(1), input_ids.size(2)
        input_ids = input_ids.view(batch_size*seq_len, input_ids.size(-1)) # input_ids:(N*S, T)
        input_masks = input_masks.view(batch_size*seq_len, input_masks.size(-1)) # input_masks(N*S, T)
        # with torch.no_grad():

        x = self.plm(input_ids, input_masks) # (N*S, T, E)
     
        x = x.view(batch_size, seq_len, token_len, -1) # (N, S, T, E)
        x, tok_batch_idxs = self.merge_sentences(x, sentence_lengths) # (N, S, E)
        
        node_masks = self.get_dialogue_masks(sentence_lengths)
        node_x = self.node_gat(x, sen_edges, tok_edges, dep_mat)

        node_x = self.reorganize(node_x, sentence_lengths)
        #node_x = torch.cat([node_x, pos_emb], dim=-1) # (N, S, 2E)
        node_logits = self.node_classifier(node_x) # (N, S, TN)
        node_logits = node_logits * node_masks.unsqueeze(-1) #(N, S, TN)
        node_logits = node_logits.transpose(1,2)
        #node_logits = torch.zeros_like(node_logits).to(node_logits.device)
       
        bs, seq_len = node_x.size(0), node_x.size(1)
        #edge_mask = self.get_edge_mask(sentence_lengths, sen_conj_mat)
   
        # # edge_mask = torch.triu(edge_mask, diagonal=1).unsqueeze(-1) #(N, S, S, 1)
        edge_x = self.edge_gat(x, sen_edges, tok_edges, dep_mat)
        edge_x = self.reorganize(edge_x, sentence_lengths)
        sen_x = self.sen_gat(x, sen_edges, tok_edges, dep_mat)
        sen_x = self.reorganize(sen_x, sentence_lengths)
        #edge_mask = self.get_edge_mask(sentence_lengths, sen_conj_mat)
        edge_mask = torch.ones((seq_len, seq_len)).to(x.device)
        edge_mask = torch.triu(edge_mask, diagonal=1).unsqueeze(0).unsqueeze(1)
        # # edge_x = edge_x * edge_mask # (N, S, S, E)
        # edge_logits = self.edge_regressioner(edge_x).squeeze(-1) # (N, S, S, 2)
        # edge_logits = F.softmax(edge_logits, dim=-1)
        edge_logits = self.edge_logiter(edge_x, token_index, thread_length)
        #edge_logits = torch.zeros_like(edge_logits).to(edge_logits.device)
        #edge_logits = edge_logits * edge_mask
        
        triplet_preds = []
        outputs = {'targets':[], 'aspects':[], 'opinions':[],
                   'target_aspect_pairs':[], 'target_opinion_pairs':[], 'aspect_opinion_pairs':[],
                   'triplets':[]}#, 'edge_mask':edge_mask}
        
        for i in range(bs):
            output = self.dataprocessor.decode_triplets_matrix(node_logits[i], edge_logits[i])
            triplet_preds.append(output['triplets'])
            for key, val in output.items():
                outputs[key].append(val)
        
        label_x, label_batch_idxs = self.extract_triplets(sen_x, triplets) # (T, 3E), T is triplet nums
        preds_x, preds_batch_idxs = self.extract_triplets(sen_x, triplet_preds)
        if label_x.size(0) > 0:
            label_x = self.polar_classifier(label_x) # (T, 3)
        if preds_x.size(0) > 0:
            preds_x = self.polar_classifier(preds_x)
        # decode a flatten list to a batch list
        triplet_labels_logits = self.get_triplets_list(label_x, triplets, label_batch_idxs, bs)
        triplet_preds_logits = self.get_triplets_list(preds_x, triplet_preds, preds_batch_idxs, bs)
        outputs['triplets'] = triplet_preds_logits

        return node_logits, edge_logits, triplet_labels_logits, outputs
    
    def reorganize(self, x:torch.Tensor,sentence_lengths):
        batch_size = len(sentence_lengths)
        sentence_length_sum = list(map(sum, sentence_lengths))
        max_sen_length = max(sentence_length_sum)
        output = x.new_zeros((batch_size, max_sen_length, self.hidden_size))
        for i in range(batch_size):
            output[i, :sentence_length_sum[i]] = x[sum(sentence_length_sum[:i]):sum(sentence_length_sum[:i+1])]
        return output
        
    
    def merge_sentences(self, x:torch.Tensor, sentence_lengths:List):
        # x:(N, S, T, E), sentece_length: list of sentence lengths
        return_tensor = []
        batch_idxs = [] #(2, S, E)
        for i in range(x.size(0)):
            batch_tensor = []
            for j, length in enumerate(sentence_lengths[i]):
                batch_tensor.append(x[i][j][:length])
                batch_idxs += [i]*length
            # batch_tensor: (S, T, E)
            batch_tensor = torch.cat(batch_tensor, dim=0)
            return_tensor.append(batch_tensor)
        output = torch.cat(return_tensor, dim=0)
        batch_idxs = torch.Tensor(batch_idxs).to(device=x.device, dtype=x.dtype)
  
        return output, batch_idxs #(L, E)
    
    def get_dialogue_masks(self, sentence_lengths:List):
        lengths = list(map(sum, sentence_lengths))
        batch_size = len(lengths)
        dialogue_masks = torch.zeros((batch_size, max(lengths)), device=self.device, requires_grad=False)
        for i, length in enumerate(lengths):
            dialogue_masks[i, :length] = 1
        for i in range(len(sentence_lengths)):
            for j in range(len(sentence_lengths[i])):
                dialogue_masks[i, sum(sentence_lengths[i][:j])] = 0
        return dialogue_masks
    
    def get_edge_mask(self, sentence_lengths:List, sen_conj_mat:torch.Tensor):
        bs = len(sentence_lengths)
        lengths = list(map(sum, sentence_lengths))
        seq_len = max(lengths)
        edge_masks = torch.zeros((bs, seq_len, seq_len), device=self.device, requires_grad=False)
        for i in range(bs):
            edge_masks[i, :lengths[i], :lengths[i]] = 1
        return edge_masks
    
    def extract_triplets(self, x:torch.Tensor, triplets:List):
        # extarct aspect, target, opinion tensors
        # x:(N, S, E)
        tensors = []
        batch_idxs = []
        for i in range(x.size(0)):
            for triplet in triplets[i]:
                #triplet = triplet.to(torch.int32)
                target_tensor = x[i][triplet[0]:triplet[1]]
                target_tensor = aggregate(target_tensor, method=self.aggregate_method) # (E)
                aspect_tensor = x[i][triplet[2]:triplet[3]]
                aspect_tensor = aggregate(aspect_tensor, method=self.aggregate_method) # (E)
                opinion_tensor = x[i][triplet[4]:triplet[5]]
                opinion_tensor = aggregate(opinion_tensor, self.aggregate_method) # (E)
                tensor = torch.concat([target_tensor, aspect_tensor, opinion_tensor])
                tensors.append(tensor)
                batch_idxs.append(i)
        if len(tensors) == 0:
            return torch.Tensor([]), [x.size(0)]
        return torch.stack(tensors), batch_idxs # (T, 3E)
    
    def get_triplets_list(self, x:torch.Tensor, triplets:torch.Tensor, batch_idxs:List, batch_size:int):
        #x: (N, S, 3)
        if x.size(0) == 0:
            return [[] for _ in range(batch_idxs[0])]
        batch_preds = [[] for _ in range(batch_size)]
        triplets = [triplets[i][j] for i in range(len(triplets)) for j in range(len(triplets[i]))]
        for i in range(x.size(0)):
            bat_idx = batch_idxs[i]
            batch_preds[bat_idx].append(torch.cat([torch.Tensor(triplets[i][:6]).to(self.device), x[i]]))
        # if len(batch_preds) == 1:
        #     print(batch_preds)
        #     print(batch_idxs)
        #     print(cur_bat_idx)
        return batch_preds
                
    def cal_node_loss(self, node_logits:torch.Tensor, node_logits_label:torch.Tensor):
        loss = F.cross_entropy(node_logits, node_logits_label, weight=torch.Tensor([1] + [self.node_loss_weight]*(self.token_label_num-1)).to(self.device))
        return loss
    
    def cal_edge_loss(self, edge_logits:torch.Tensor, edge_logits_label:torch.Tensor):
        # edge_logits : (N, C, S, S) # edge_logits_label: (N, C, S, S)
        loss = F.cross_entropy(edge_logits, edge_logits_label, weight=torch.Tensor([1] + [self.edge_loss_weight]*(self.edge_classes-1)).to(self.device))
        return loss
    
    
    def cal_polar_loss(self, triplet_logits:List, triplet_labels:List):
        logits, labels = [], []
        logits = [triplet_logits[i][j][-3:] for i in range(len(triplet_logits)) for j in range(len(triplet_logits[i]))]
        labels = [triplet_labels[i][j][6] for i in range(len(triplet_labels)) for j in range(len(triplet_labels[i]))]
        if len(logits) == 0 and len(labels) == 0:
            return torch.Tensor([0]).to(self.device)
        logits = torch.stack(logits)
        labels = torch.stack(labels).to(torch.int64)
        labels = F.one_hot(labels, num_classes=4)[:,1:].float()
        return F.cross_entropy(logits, labels)
        
    def on_train_epoch_end(self):
        metrics = self.train_metric.compute()
        metrics = {'train_'+key:val for key, val in metrics.items()}
        self.log_dict(metrics)
        
    def on_validation_epoch_end(self):
        metrics = self.valid_metric.compute()
        metrics = {'valid_'+key:val for key, val in metrics.items()}
        self.log_dict(metrics)
        
    def on_test_epoch_end(self):
        metrics = self.test_metric.compute()
        metrics = {'test_'+key:val for key, val in metrics.items()}
        self.log_dict(metrics)    
        
            
                        
                        
        