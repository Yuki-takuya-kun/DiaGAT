from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
import torchmetrics.utilities as utils

from src.common import DataProcessor

class DiaMetric(Metric):
    def __init__(self,
                 dataprocessor:DataProcessor,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataprocessor = dataprocessor
        self.add_state("target_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("aspect_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("opinion_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("opinion_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("opinion_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("target_aspect_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_aspect_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_aspect_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("target_opinion_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_opinion_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_opinion_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("aspect_opinion_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_opinion_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_opinion_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')

        self.add_state("quad_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("quad_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("quad_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("iden_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("iden_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("iden_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("intra_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("intra_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("intra_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("inter_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("inter_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("inter_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("cross1_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross1_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross1_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("cross2_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross2_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross2_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("cross3_tp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross3_fp", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("cross3_fn", default=torch.Tensor([0]), dist_reduce_fx='sum')
        
        self.add_state("target_fp_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_fp_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("edge_fp_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("target_fn_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("aspect_fn_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
        self.add_state("edge_fn_num", default=torch.Tensor([0]), dist_reduce_fx='sum')
    
    def update(self, preds:Dict, entity_targets:torch.Tensor,
                relation_targets:torch.Tensor,
                pair_targets: List,
                triplet_targets:List,
               sentence_lengths:List,
               cross_mats:List,
               sentence_ids:List) -> None:
        token2senid = sentence_ids
        batch_target_preds, batch_aspect_preds, batch_opinion_preds = preds['targets'], preds['aspects'], preds['opinions']
        batch_target_aspect_preds, batch_target_opinion_preds, batch_aspect_opinion_preds = preds['target_aspect_pairs'], \
            preds['target_opinion_pairs'], preds['aspect_opinion_pairs']
        batch_triplet_preds = preds['triplets']
        
        for i in range(len(batch_target_preds)):
            target_preds, aspect_preds, opinion_preds = batch_target_preds[i], batch_aspect_preds[i], batch_opinion_preds[i]
            target_labels, aspect_labels, opinion_labels = self.dataprocessor.decode_entity_matrix(entity_targets[i])
            
            self.target_tp += len(set(target_preds) & set(target_labels))
            self.target_fp += len(set(target_preds) - set(target_labels))
            self.target_fn += len(set(target_labels) - set(target_preds))
            
            self.aspect_tp += len(set(aspect_preds) & set(aspect_labels))
            self.aspect_fp += len(set(aspect_preds) - set(aspect_labels))
            self.aspect_fn += len(set(aspect_labels) - set(aspect_preds))
            
            self.opinion_tp += len(set(opinion_preds) & set(opinion_labels))
            self.opinion_fp += len(set(opinion_preds) - set(opinion_labels))
            self.opinion_fn += len(set(opinion_labels) - set(opinion_preds))
            
            target = f"target: {int(self.target_tp)}  {int(self.target_fp)}  {int(self.target_fn)}"
            aspect = f"aspect: {int(self.aspect_tp)}  {int(self.aspect_fp)}  {int(self.aspect_fn)}"
            opinion = f"opinion: {int(self.opinion_tp)} {int(self.opinion_fp)} {int(self.opinion_fn)}"
            #tqdm.write(target + " "*(30-len(target)) + " | " + aspect + " "*(30-len(aspect)) + " | " + opinion)
            
            target_aspect_preds = batch_target_aspect_preds[i]
            target_aspect_labels = pair_targets[i]['ta']
            target_opinion_preds = batch_target_opinion_preds[i]
            target_opinion_labels = pair_targets[i]['to']
            aspect_opinion_preds = batch_aspect_opinion_preds[i]
            aspect_opinion_labels = pair_targets[i]['ao']
            
            self.target_aspect_tp += len(set(target_aspect_preds) & set(target_aspect_labels))
            self.target_aspect_fp += len(set(target_aspect_preds) - set(target_aspect_labels))
            self.target_aspect_fn += len(set(target_aspect_labels) - set(target_aspect_preds))
            
            ta_fp_set = set(target_aspect_preds) - set(target_aspect_labels)
            for pair in ta_fp_set:
                t = (pair[0], pair[1])
                a = (pair[2], pair[3])
                if t not in set(target_labels):
                    self.target_fp_num += 1
                elif a not in set(aspect_labels):
                    self.aspect_fp_num += 1
                else:
                    self.edge_fp_num += 1
            #tqdm.write(f"target_fp_num: {float(self.target_fp_num)}    aspect_fp_num:{float(self.aspect_fp_num)}  edge_fp_num: {float(self.edge_fp_num)}")
            
            ta_fn_set = set(target_aspect_labels) - set(target_aspect_preds)
            for pair in ta_fn_set:
                t = (pair[0], pair[1])
                a = (pair[2], pair[3])
                if t not in set(target_preds):
                    self.target_fn_num += 1
                elif a not in set(aspect_preds):
                    self.aspect_fn_num += 1
                else:
                    ##tqdm.write(f"fn_edge: {t, a}")
                    self.edge_fn_num += 1
            #tqdm.write(f"target_fn_num: {int(self.target_fn_num)}     aspect_fn_num:{int(self.aspect_fn_num)}  edge_fn_num:{int(self.edge_fn_num)}")
            
            self.target_opinion_tp += len(set(target_opinion_preds) & set(target_opinion_labels))
            self.target_opinion_fp += len(set(target_opinion_preds) - set(target_opinion_labels))
            self.target_opinion_fn += len(set(target_opinion_labels) - set(target_opinion_preds))
            
            self.aspect_opinion_tp += len(set(aspect_opinion_preds) & set(aspect_opinion_labels))
            self.aspect_opinion_fp += len(set(aspect_opinion_preds) - set(aspect_opinion_labels))
            self.aspect_opinion_fn += len(set(aspect_opinion_labels) - set(aspect_opinion_preds))
            ta = f"ta: {int(self.target_aspect_tp)}  {int(self.target_aspect_fp)}  {int(self.target_aspect_fn)}"
            to = f"to: {int(self.target_opinion_tp)}  {int(self.target_opinion_fp)}  {int(self.target_opinion_fn)}"
            ao = f"ao: {int(self.aspect_opinion_tp)}  {int(self.aspect_opinion_fp)}  {int(self.aspect_opinion_fn)}"
            #tqdm.write(ta + " "*(30-len(ta)) + " | " + to + " "*(30-len(to)) + " | " + ao)
        
            triplet_preds = [triplet.detach().cpu().tolist() for triplet in batch_triplet_preds[i]]
            triplet_preds = [tuple([int(num) for num in elem[:6]] + [int(np.argmax(elem[-3:]))+1]) for elem in triplet_preds]
            triplet_labels = [tuple(triplet.detach().cpu().tolist()) for triplet in triplet_targets[i]]
            quad_tp_set = set(triplet_preds) & set(triplet_labels)
            quad_fp_set = set(triplet_preds) - set(triplet_labels)
            quad_fn_set = set(triplet_labels) - set(triplet_preds)
            self.quad_tp += len(quad_tp_set)
            self.quad_fp += len(quad_fp_set)
            self.quad_fn += len(quad_fn_set)
            
            
            iden_triplet_preds = [triplet[:6] for triplet in triplet_preds]
            iden_triplet_labels = [triplet[:6] for triplet in triplet_labels]
            iden_tp_set = set(iden_triplet_preds) & set(iden_triplet_labels)
            iden_fp_set = set(iden_triplet_preds) - set(iden_triplet_labels)
            iden_fn_set = set(iden_triplet_labels) - set(iden_triplet_preds)
            self.iden_tp += len(iden_tp_set)
            self.iden_fp += len(iden_fp_set)
            self.iden_fn += len(iden_fn_set)
            
            quad = f"quad: {int(self.quad_tp)}  {int(self.quad_fp)}  {int(self.quad_fn)}"
            iden = f"iden: {int(self.iden_tp)}  {int(self.iden_fp)}  {int(self.iden_fn)}"
            #tqdm.write(quad + " "*(30-len(quad)) + " | " + iden + " "*(30-len(iden)))
            
            intra_preds, inter_preds, cross1_preds, cross2_preds, cross3_preds = \
                self.get_cross_lists(triplet_preds, cross_mats[i], token2senid[i])
            intra_labels, inter_labels, cross1_labels, cross2_labels, cross3_labels = \
                self.get_cross_lists(triplet_labels, cross_mats[i], token2senid[i])
          
            self.intra_tp += len(intra_preds & intra_labels)
            self.intra_fp += len(intra_preds - intra_labels)
            self.intra_fn += len(intra_labels - intra_preds)
            
            self.inter_tp += len(inter_preds & inter_labels)
            self.inter_fp += len(inter_preds - inter_labels)
            self.inter_fn += len(inter_labels - inter_preds)
            
            self.cross1_tp += len(cross1_preds & cross1_labels)
            self.cross1_fp += len(cross1_preds - cross1_labels)
            self.cross1_fn += len(cross1_labels - cross1_preds)
            
            self.cross2_tp += len(cross2_preds & cross2_labels)
            self.cross2_fp += len(cross2_preds - cross2_labels)
            self.cross2_fn += len(cross2_labels - cross2_preds)
            
            self.cross3_tp += len(cross3_preds & cross3_labels)
            self.cross3_fp += len(cross3_preds - cross3_labels)
            self.cross3_fn += len(cross3_labels - cross3_preds)
            # quad_tp_cross = [[token2senid[i][idx] for idx in triplet[:6:2]] for triplet in quad_tp_set]
            # for triplet_cross in quad_tp_set:
            #     ta_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[1])]
            #     to_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[2])]
            #     ao_cross = cross_mats[i][int(triplet_cross[1]), int(triplet_cross[2])]
            #     cross = max(ta_cross, to_cross, ao_cross)
            #     if cross == 0:
            #         self.intra_tp += 1
            #     else:
            #         self.inter_tp += 1
            #         if cross == 1:
            #             self.cross1_tp += 1
            #         elif cross == 2:
            #             self.cross2_tp += 1
            #         elif cross >= 3:
            #             self.cross3_tp += 1
            #         else:
            #             raise ValueError(f"cross error, get {cross}")
            
            # quad_fp_cross = [[token2senid[i][idx] for idx in triplet[:6:2]] for triplet in quad_fp_set]
            # for triplet_cross in quad_fp_cross:
            #     ta_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[1])]
            #     to_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[2])]
            #     ao_cross = cross_mats[i][int(triplet_cross[1]), int(triplet_cross[2])]
            #     cross = max(ta_cross, to_cross, ao_cross)
            #     if cross == 0:
            #         self.intra_fp += 1
            #     else:
            #         self.inter_fp += 1
            #         if cross == 1:
            #             self.cross1_fp += 1
            #         elif cross == 2:
            #             self.cross2_fp += 1
            #         elif cross >= 3:
            #             self.cross3_fp += 1
            #         else:
            #             raise ValueError(f"cross error, get {cross}")
                    
            # quad_fn_cross = [[token2senid[i][idx] for idx in triplet[:6:2]] for triplet in quad_fn_set]
            # for triplet_cross in quad_fn_cross:
            #     ta_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[1])]
            #     to_cross = cross_mats[i][int(triplet_cross[0]), int(triplet_cross[2])]
            #     ao_cross = cross_mats[i][int(triplet_cross[1]), int(triplet_cross[2])]
            #     cross = max(ta_cross, to_cross, ao_cross)
            #     if cross == 0:
            #         self.intra_fn += 1
            #     else:
            #         self.inter_fn += 1
            #         if cross == 1:
            #             self.cross1_fn += 1
            #         elif cross == 2:
            #             self.cross2_fn += 1
            #         elif cross >= 3:
            #             self.cross3_fn += 1
            #         else:
            #             raise ValueError(f"cross error, get {cross}")
            intra = f"intra: {int(self.intra_tp)}  {int(self.intra_fp)}  {int(self.intra_fn)}"
            inter = f"inter: {int(self.inter_tp)}  {int(self.inter_fp)}  {int(self.inter_fn)}"
            #tqdm.write(intra + " "*(30-len(intra)) + " | " + inter + " "*(30-len(inter)))
            cross1 = f"cross1: {int(self.cross1_tp)}  {int(self.cross1_fp)}  {int(self.cross1_fn)}"
            cross2 = f"cross2: {int(self.cross2_tp)}  {int(self.cross2_fp)}  {int(self.cross2_fn)}"
            cross3 = f"cross3: {int(self.cross3_tp)}  {int(self.cross3_fp)}  {int(self.cross3_fn)}"
            #tqdm.write(cross1 + " "*(30-len(cross1)) + " | " + cross2 + " "*(30-len(cross2)) + " | " + cross3)
            #tqdm.write("-"*30+'-+-'+'-'*30+'-+-' + '-'*30)
    
    def get_cross_lists(self, triplets, cross_mat, token2senid):
        intra, inter, cross1, cross2, cross3 = [], [], [], [], []
        for triplet in triplets:        
            target_idx = token2senid[int(triplet[0])]
            aspect_idx = token2senid[int(triplet[2])]
            opinion_idx = token2senid[int(triplet[4])]
            ta_cross = cross_mat[target_idx, aspect_idx]
            to_cross = cross_mat[target_idx, opinion_idx]
            ao_cross = cross_mat[aspect_idx, opinion_idx]
            cross = max(ta_cross, to_cross, ao_cross)
            if cross == 0:
                intra.append(triplet)
            else:
                inter.append(triplet)
                if cross == 1:
                    cross1.append(triplet)
                elif cross == 2:
                    cross2.append(triplet)
                elif cross >= 3:
                    cross3.append(triplet)
                else:
                    raise ValueError
        return set(intra), set(inter), set(cross1), set(cross2), set(cross3)
                
    def get_token2senid(self, sentence_lengths:List) -> List[torch.Tensor]:
        token2senid = []
        for i in range(len(sentence_lengths)):
            batch_sentence_length = torch.cat([torch.IntTensor(0), torch.IntTensor(sentence_lengths[i])])
            batch_token2sen = torch.zeros(sum(sentence_lengths[i]), dtype=int)
            for j in range(1, len(batch_sentence_length)):
                batch_token2sen[batch_sentence_length[j-1]:batch_sentence_length[j]] = j
            token2senid.append(batch_token2sen)
        return token2senid
                        
    def compute(self) -> Dict:
        target_precision = self.target_tp / (self.target_tp + self.target_fp + 1e-6)
        target_recall = self.target_tp / (self.target_tp + self.target_fn + 1e-6)
        target_f1 = 2 * target_precision * target_recall / (target_precision + target_recall + 1e-6)
        
        aspect_precision = self.aspect_tp / (self.aspect_tp + self.aspect_fp + 1e-6)
        aspect_recall = self.aspect_tp / (self.aspect_tp + self.aspect_fn + 1e-6)
        aspect_f1 = 2 * aspect_precision * aspect_recall / (aspect_precision + aspect_recall + 1e-6)
        
        opinion_precision = self.opinion_tp / (self.opinion_tp + self.opinion_fp + 1e-6)
        opinion_recall = self.opinion_tp / (self.opinion_tp + self.opinion_fn + 1e-6)
        opinion_f1 = 2 * opinion_precision * opinion_recall / (opinion_precision + opinion_recall + 1e-6)
        
        target_aspect_precision = self.target_aspect_tp / (self.target_aspect_tp + self.target_aspect_fp + 1e-6)
        target_aspect_recall = self.target_aspect_tp / (self.target_aspect_tp + self.target_aspect_fn + 1e-6)
        target_aspect_f1 = 2 * target_aspect_precision * target_aspect_recall / (target_aspect_precision + target_aspect_recall + 1e-6)
            
        target_opinion_precision = self.target_opinion_tp / (self.target_opinion_tp + self.target_opinion_fp + 1e-6)
        target_opinion_recall = self.target_opinion_tp / (self.target_opinion_tp + self.target_opinion_fn + 1e-6)
        target_opinion_f1 = 2 * target_opinion_precision * target_opinion_recall / (target_opinion_precision + target_opinion_recall + 1e-6)
            
        aspect_opinion_precision = self.aspect_opinion_tp / (self.aspect_opinion_tp + self.aspect_opinion_fp + 1e-6)
        aspect_opinion_recall = self.aspect_opinion_tp / (self.aspect_opinion_tp + self.aspect_opinion_fn + 1e-6)
        aspect_opinion_f1 = 2 * aspect_opinion_precision * aspect_opinion_recall / (aspect_opinion_precision + aspect_opinion_recall + 1e-6)
            
        quad_precision = self.quad_tp / (self.quad_tp + self.quad_fp + 1e-6)
        quad_recall = self.quad_tp / (self.quad_tp + self.quad_fn + 1e-6)
        quad_f1 = 2 * quad_precision * quad_recall / (quad_precision + quad_recall + 1e-6)
        
        iden_precision = self.iden_tp / (self.iden_tp + self.iden_fp + 1e-6)
        iden_recall = self.iden_tp / (self.iden_tp + self.iden_fn + 1e-6)
        iden_f1 = 2 * iden_precision * iden_recall / (iden_precision + iden_recall + 1e-6)
        
        intra_precision = self.intra_tp / (self.intra_tp + self.intra_fp + 1e-6)
        intra_recall = self.intra_tp / (self.intra_tp + self.intra_fn + 1e-6)
        intra_f1 = 2 * intra_precision * intra_recall / (intra_precision + intra_recall + 1e-6)
        
        inter_precision = self.inter_tp / (self.inter_tp + self.inter_fp + 1e-6)
        inter_recall = self.inter_tp / (self.inter_tp + self.inter_fn + 1e-6)
        inter_f1 = 2 * inter_precision * inter_recall / (inter_precision + inter_recall + 1e-6)
        
        cross1_precision = self.cross1_tp / (self.cross1_tp + self.cross1_fp + 1e-6)
        cross1_recall = self.cross1_tp / (self.cross1_tp + self.cross1_fn + 1e-6)
        cross1_f1 = 2 * cross1_precision * cross1_recall / (cross1_precision + cross1_recall + 1e-6)
            
        cross2_precision = self.cross2_tp / (self.cross2_tp + self.cross2_fp + 1e-6)
        cross2_recall = self.cross2_tp / (self.cross2_tp + self.cross2_fn + 1e-6)
        cross2_f1 = 2 * cross2_precision * cross2_recall / (cross2_precision + cross2_recall + 1e-6)
            
        cross3_precision = self.cross3_tp / (self.cross3_tp + self.cross3_fp + 1e-6)
        cross3_recall = self.cross3_tp / (self.cross3_tp + self.cross3_fn + 1e-6)
        cross3_f1 = 2 * cross3_precision * cross3_recall / (cross3_precision + cross3_recall + 1e-6)
        
        self.reset()
        
        return {
            'target_precision':target_precision,
            'target_recall': target_recall,
            'target_f1': target_f1,
            'aspect_precision': aspect_precision,
            'aspect_recall': aspect_recall,
            'aspect_f1': aspect_f1,
            'opinion_precision': opinion_precision,
            'opinion_recall': opinion_recall,
            'opinion_f1': opinion_f1,
            'target_aspect_precision': target_aspect_precision,
            'target_aspect_recall': target_aspect_recall,
            'target_aspect_f1': target_aspect_f1,
            'target_opinion_precision': target_opinion_precision,
            'target_opinion_recall': target_opinion_recall,
            'target_opinion_f1': target_opinion_f1,
            'aspect_opinion_precision': aspect_opinion_precision,
            'aspect_opinion_recall': aspect_opinion_recall,
            'aspect_opinion_f1': aspect_opinion_f1,
            'quad_precision': quad_precision,
            'quad_recall': quad_recall,
            'quad_f1': quad_f1,
            'iden_precision': iden_precision,
            'iden_recall': iden_recall,
            'iden_f1': iden_f1,
            'intra_precision': intra_precision,
            'intra_recall': intra_recall,
            'intra_f1': intra_f1,
            'inter_precision': inter_precision,
            'inter_recall': inter_recall,
            'inter_f1': inter_f1,
            'cross1_precision': cross1_precision,
            'cross1_recall': cross1_recall,
            'cross1_f1': cross1_f1,
            'cross2_precision': cross2_precision,
            'cross2_recall': cross2_recall,
            'cross2_f1': cross2_f1,
            'cross3_precision': cross3_precision,
            'cross3_recall': cross3_recall,
            'cross3_f1': cross3_f1
        }

        
        