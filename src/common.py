import os
import random
import math
from typing import Union
from collections import defaultdict
from itertools import permutations, accumulate, product
from bidict import bidict
from tqdm import tqdm

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ltp import LTP
import stanza
from stanza.pipeline.core import DownloadMethod

class WordPair:
    def __init__(self, lang):
        self.lang = lang
        # 'O' denotes that the element is None
        self.entity_dic = bidict({"O": 0, 
                                  "B-T":1, "I-T":2, "E-T":3, "S-T":4,
                                  "B-A":5, "I-A":6, "E-A":7, "S-A":8,
                                  "B-O":9, "I-O":10, "E-O":11, "S-O":12})

        self.polarity_dic = bidict({"O": 0, "pos": 1, "neg": 2, 'other': 3})
        self.relation_dic = bidict({'O':0, 'h2h':1, 't2t':2, 's':3})
        
        self.ltp_pos_dict = bidict({
            'pad':0, 'b':1, 'c':2, 'd':3, 'e':4, 'g':5, 'h':6, 'i':7, 'j':8,
            'k':9, 'm':10, 'n':11, 'nd':12, 'nh':13, 'ni':14, 'nl':15, 'ns':16,
            'nt':17, 'nz':18, 'o':19, 'p':20, 'q':21, 'r':22, 'u':23, 'v':24,
            'wp':25, 'ws':26, 'x':27, 'z':28, 'a':29, 'head':30
        })
        
        self.ltp_dep_dict = bidict({
            'PAD':0, 'VOB':1, 'IOB':2, 'FOB':3, 'DBL':4, 'ATT':5, 'ADV':6,
            'CMP':7, 'COO':8, 'POB':9, 'LAD':10, 'RAD':11, 'IS':12, 'HED':13,
            'WP':14, 'ST':15, 'SELF':16, 'SBV':17
        })
        
        self.stanza_pos_dict = bidict({
            'PAD':0, 'ADP':1, 'ADV':2, 'AUX':3, 'CCONJ':4, 'DET':5,
            'INTJ':6, 'NOUN':7, 'NUM':8, 'PART':9, 'PRON':10, 'PROPN':11,
            'PUNCT':12, 'SCONJ':13, 'SYM':14, 'VERB':15, 'X':16, 'root':17,
            'ADJ': 18
        })
        
        self.stanza_dep_dict = bidict({
            'pad':0, 'acl:relcl':1, 'advcl':2, 'advcl:relcl':3, 'advmod':4,
            'advmod:emph':5, 'advmod:lmod':6, 'amod':7, 'appos':8, 'aux':9,
            'aux:pass':10, 'case':11, 'cc':12, 'cc:preconj':13, 'ccomp':14,
            'clf':15, 'compound':16, 'compound:lvc':17, 'compound:prt':18,
            'compound:redup':19, 'compound:svc':20, 'conj':21, 'cop':22,
            'csubj':23, 'csubj:outer':24, 'csubj:pass':25, 'dep':26,
            'det':27, 'det:numgov':28, 'det:nummod':29, 'det:poss':30,
            'det:predet':31, 'discourse':32, 'dislocated':33, 'expl':34,
            'expl:impers':35,'expl:pass':36, 'expl:pv':37, 'fixed':38, 'flat':39,
            'flat:foreign':40, 'flat:name':41, 'goeswith':42, 'iobj':43,
            'list':44, 'mark':45, 'nmod':46, 'nmod:poss':47, 'nmod:tmod':48,
            'nmod:npmod':49, 'nsubj':50, 'nsubj:outer':51, 'nsubj:pass':52, 'nummod':53,
            'nummod:gov':54, 'obj':55, 'obl':56, 'obl:agent':57, 'obl:arg':58,
            'obl:lmod':59, 'obl:npmod':60, 'obl:tmod':61, 'orphan':62, 'parataxis':63,
            'punct':64, 'reparandum':65, 'root':66, 'vocative':67, 'xcomp':68,
            'st':69, 'self':70, 'acl':71, 'compound:ext':72, 'mark:rel':73,
            'mark:adv':74
        })
        
    @property
    def entity_num(self): return len(self.entity_dic)
    
    @property
    def polarity_num(self): return len(self.polarity_dic)
    
    @property
    def relation_num(self): return len(self.relation_dic)
    
    @property
    def pos_num(self):
        if self.lang == 'zh': return len(self.ltp_pos_dict)
        elif self.lang == 'en': return len(self.stanza_pos_dict)
        else: raise ValueError(f'{self.lang} is not support')
    
    @property
    def dep_num(self):
        if self.lang == 'zh': return len(self.ltp_dep_dict)
        elif self.lang == 'en': return len(self.stanza_dep_dict)
        else: raise ValueError(f'{self.lang} is not support')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_gaussian_matrix(shape, sigma):
    """
    Generate a small matrix with Gaussian distribution.
    The peak of the Gaussian distribution is centered and the highest value is 1.
    :param shape: Shape of the matrix
    :return: Generated matrix
    """
    # Create a meshgrid for the matrix
    x, y = np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0]))

    # Calculate the distance from the center
    d = np.sqrt(x**2 + y**2)

    # Gaussian distribution
    gaussian_matrix = np.exp(-(d**2 / (2.0 * sigma**2)))

    # Normalize to make the highest value 1
    gaussian_matrix /= np.max(gaussian_matrix)

    return gaussian_matrix

    
class DataProcessor:
    def __init__(self, cfg, stanza_model=None, ltp_model=None):
        self.cfg = cfg
        self.wordpair = WordPair(self.cfg.lang)
        self.entity_dic = self.wordpair.entity_dic
        self.stanza = stanza_model
        self.ltp = ltp_model
        # if stanza_model is None and ltp_model is None:
        #     if self.cfg.lang == 'zh':
        #         self.ltp = LTP()
        #     elif self.cfg.lang == 'en':
        #         self.stanza_tokenize = stanza.Pipeline(lang='en', dir=self.cfg.stanza_dir, processors='tokenize', tokenize_no_ssplit=True, download_method=DownloadMethod.NONE)
        #         self.stanza = stanza.Pipeline(lang='en', dir=self.cfg.stanza_dir, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, download_method=DownloadMethod.NONE)
        self.sigma = 0.8
        self.threshold = 0.8
    
    def encode_entity_matrix(self, targets, aspects, opinions, seq_len):
        """
        encode entity list into matrix
        """
        matrix = np.zeros((len(self.entity_dic), seq_len))
        for s, e, text in targets:
            if s == e-1:
                matrix[self.entity_dic.get('S-T'), s] = 1
            else:
                matrix[self.entity_dic.get('B-T'), s] = 1
                matrix[self.entity_dic.get('I-T'), s+1:e-1] = 1
                matrix[self.entity_dic.get('E-T'), e-1] = 1
        for s, e, text in aspects:
            if s == e-1:
                matrix[self.entity_dic.get('S-A'), s] = 1
            else:
                matrix[self.entity_dic.get('B-A'), s] = 1
                matrix[self.entity_dic.get('I-A'), s+1:e-1] = 1
                matrix[self.entity_dic.get('E-A'), e-1] = 1
        for s, e, text, p in opinions:
            if s == e-1:
                matrix[self.entity_dic.get('S-O'), s] = 1
            else:
                matrix[self.entity_dic.get('B-O'), s] = 1
                matrix[self.entity_dic.get('I-O'), s+1:e-1] = 1
                matrix[self.entity_dic.get('E-O'), e-1] = 1
        matrix[self.entity_dic.get('O')] = (np.sum(matrix, axis=0) == 0).astype(np.int64)
        return matrix
    
    def decode_entity_matrix(self, entity_matrix:Union[torch.Tensor, np.ndarray]):
        """
        decode entity matrix into targets, aspects, opinions lists
        Parameters:
            entity_matrix: the entity matrix that should be decoded.
        """
        if isinstance(entity_matrix, torch.Tensor):
            codes = torch.argmax(entity_matrix, dim=0).detach().cpu().tolist()
        if isinstance(entity_matrix, np.ndarray):
            codes = np.argmax(entity_matrix, axis=0).tolist()
        targets, aspects, opinions = [], [], []
        start_idx, i = 0, 0
        # extract target aspect and opinion pairs
        while i < len(codes):
            code0 = codes[i]
            flag = False
            if code0 == self.entity_dic['S-T']:
                targets.append((i, i+1))
                i += 1
            elif code0 == self.entity_dic['S-A']:
                aspects.append((i, i+1))
                i += 1
            elif code0 == self.entity_dic['S-O']:
                opinions.append((i, i+1))
                i += 1
            elif code0 in [self.entity_dic['B-T'], self.entity_dic['B-A'], self.entity_dic['B-O']]:
                start_idx = i
                for j in range(i+1, len(codes)):
                    code1 = codes[j]
                    
                    if code0 == self.entity_dic['B-T'] and code1 == self.entity_dic['I-T'] or \
                        code0 == self.entity_dic['B-A'] and code1 == self.entity_dic['I-A'] or \
                        code0 == self.entity_dic['B-O'] and code1 == self.entity_dic['I-O']:
                        continue
                    elif code0 == self.entity_dic['B-T'] and code1 == self.entity_dic['E-T']:
                        targets.append((start_idx, j+1))
                        i, flag = j+1, True
                        break
                    elif code0 == self.entity_dic['B-A'] and code1 == self.entity_dic['E-A']:
                        aspects.append((start_idx, j+1))
                        i, flag = j+1, True
                        break
                    elif code0 == self.entity_dic['B-O'] and code1 == self.entity_dic['E-O']:
                        opinions.append((start_idx, j+1))
                        i, flag = j+1, True
                        break
                    else:
                        break
                if not flag: i += 1
            else: i += 1
        return targets, aspects, opinions
    
    def entity_code2str(self, codes):
        """
        transfer entity code into string tag
        """
        return [self.entity_dic.inverse.get(code) for code in codes]
    
    def encode_relation_matrix(self, triplets, seq_len):
        """
        encoder relations into probability space
        """
        matrix = np.zeros((self.wordpair.relation_num, seq_len, seq_len), np.float32)
        matrix[self.wordpair.relation_dic['O']] = 1
        matrix[0] = np.triu(matrix[0], k=1)
        for t_s, t_e, a_s, a_e, o_s, o_e, polaries, t_t, a_t, o_t in triplets:
            if t_e - t_s > 0 and a_e - a_s > 0:
                if t_e - t_s == 1 and a_e - a_s == 1:
                    if t_s < a_s:
                        matrix[self.wordpair.relation_dic['O'], t_s, a_s] = 0
                        matrix[self.wordpair.relation_dic['s'], t_s, a_s] = 1
                    else:
                        matrix[self.wordpair.relation_dic['O'], a_s, t_s] = 0
                        matrix[self.wordpair.relation_dic['s'], a_s, t_s] = 1
                elif t_s < a_s:
                    matrix[self.wordpair.relation_dic['O'], t_s, a_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], t_s, a_s] = 1
                    matrix[self.wordpair.relation_dic['O'], t_e-1, a_e-1] = 0
                    matrix[self.wordpair.relation_dic['t2t'], t_e-1, a_e-1] = 1
                else:
                    matrix[self.wordpair.relation_dic['O'], a_s, t_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], a_s, t_s] = 1
                    matrix[self.wordpair.relation_dic['O'], a_e-1, t_e-1] = 0
                    matrix[self.wordpair.relation_dic['t2t'], a_e-1, t_e-1] = 1
            else: raise ValueError
            
            if t_e - t_s > 0 and o_e - o_s > 0:
                if t_e - t_s == 1 and o_e - o_s == 1:
                    if t_s < o_s:
                        matrix[self.wordpair.relation_dic['O'], t_s, o_s] = 0
                        matrix[self.wordpair.relation_dic['s'], t_s, o_s] = 1
                    else:
                        matrix[self.wordpair.relation_dic['O'], o_s, t_s] = 0
                        matrix[self.wordpair.relation_dic['s'], o_s, t_s] = 1
                elif t_s < o_s:
                    matrix[self.wordpair.relation_dic['O'], t_s, o_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], t_s, o_s] = 1
                    matrix[self.wordpair.relation_dic['O'], t_e-1, o_e-1] = 0
                    matrix[self.wordpair.relation_dic['t2t'], t_e-1, o_e-1] = 1
                else:
                    matrix[self.wordpair.relation_dic['O'], o_s, t_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], o_s, t_s] = 1
                    matrix[self.wordpair.relation_dic['O'], o_e-1, t_e-1] = 0
                    matrix[self.wordpair.relation_dic['t2t'], o_e-1, t_e-1] = 1
            else: raise ValueError
            
            if a_e - a_s > 0 and o_e - o_s > 0:
                if a_e - a_s == 1 and o_e - o_s == 1:
                    if a_s < o_s:
                        matrix[self.wordpair.relation_dic['O'], a_s, o_s] = 0
                        matrix[self.wordpair.relation_dic['s'], a_s, o_s] = 1
                    else:
                        matrix[self.wordpair.relation_dic['O'], o_s, a_s] = 0
                        matrix[self.wordpair.relation_dic['s'], o_s, a_s] = 1
                elif a_s < o_s:
                    matrix[self.wordpair.relation_dic['O'], a_s, o_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], a_s, o_s] = 1
                    matrix[self.wordpair.relation_dic['O'], a_e-1, o_e-1] = 0
                    matrix[self.wordpair.relation_dic['t2t'], a_e-1, o_e-1] = 1
                else:
                    matrix[self.wordpair.relation_dic['O'], o_s, a_s] = 0
                    matrix[self.wordpair.relation_dic['h2h'], o_s, a_s] = 1
                    matrix[self.wordpair.relation_dic['O'], o_e-1, a_e-1] = 0 
                    matrix[self.wordpair.relation_dic['t2t'], o_e-1, a_e-1] = 1 
            else: raise ValueError
        #print(matrix)
        triu = np.stack([np.triu(matrix[i], k=1) for i in range(matrix.shape[0])])
        #print(triu)
        assert np.array_equal(matrix, triu)
        return matrix
    
    def encode_speakers(self, speakers):
        """
        encoder speaker list into speaker mask
        """
        seq_len = len(speakers)
        matrix = np.zeros((seq_len, seq_len))
        for i, speaker_i in enumerate(speakers):
            for j, speaker_j in enumerate(speakers):
                if speaker_i == speaker_j:
                    matrix[i, j] = 1
        return matrix
    
    def decode_relation_matrix(self, targets, aspects, opinions, relation_matrix:Union[torch.Tensor, np.ndarray], check_dis:bool=False):
        """
        Decode relation matrix into pairs and triplets
        Parameters:
            entity_list: the entity list of code
            relation_matrix: relation_matrix that will be decoded
            check_dis: check the realtion matrix if is a gaussian distribution whlie generating dataset
        """
        if isinstance(relation_matrix, torch.Tensor):
            relation_matrix = relation_matrix.detach().cpu().numpy()
        
        #decode pairs
        target_aspect_pairs = self.decode_pairs(targets, aspects, relation_matrix, check_dis)
        target_opinion_pairs = self.decode_pairs(targets, opinions, relation_matrix, check_dis)
        aspect_opinion_pairs = self.decode_pairs(aspects, opinions, relation_matrix, check_dis)
        triplets = self.decode_triplets(target_aspect_pairs, target_opinion_pairs, aspect_opinion_pairs)
        return target_aspect_pairs, target_opinion_pairs, aspect_opinion_pairs, triplets
    
    def decode_pairs(self, entities1, entities2, relation_matrix, check_dis:bool=False):
        if isinstance(relation_matrix, torch.Tensor):
            relation_matrix = relation_matrix.detach().cpu().numpy()
        pairs = []
        pair_combinations = product(entities1, entities2)
        for elem in pair_combinations:
            s1, e1, s2, e2 = elem[0][0], elem[0][1], elem[1][0], elem[1][1]
            if e1 - s1 == 1 and e2 - s2 == 1:
                if s1 < s2:
                    s_label = np.argmax(relation_matrix[:, s1, s2])
                else:
                    #print(relation_matrix.shape)
                    s_label = np.argmax(relation_matrix[:, s2, s1])
                if s_label == self.wordpair.relation_dic['s']:
                    pairs.append((s1, e1, s2, e2))
            else:
                if s1 < s2:
                    h2h_label = np.argmax(relation_matrix[:, s1, s2])
                    t2t_label = np.argmax(relation_matrix[:, e1-1, e2-1])
                else:
                    h2h_label = np.argmax(relation_matrix[:, s2, s1])
                    t2t_label = np.argmax(relation_matrix[:, e2-1, e1-1])

                if h2h_label == self.wordpair.relation_dic['h2h'] and t2t_label == self.wordpair.relation_dic['t2t']:
                    pairs.append((s1, e1, s2, e2))
            
            # if p >= self.threshold:
            #     if check_dis:
            #         if s1 < s2:
            #             gaussian_matrix = generate_gaussian_matrix((e1-s1, e2-s2), self.sigma)
            #             assert np.allclose(gaussian_matrix, relation_matrix[s1:e1, s2:e2])
            #         else:
            #             gaussian_matrix = generate_gaussian_matrix((e2-s2, e1-s1), self.sigma)
            #             assert np.allclose(gaussian_matrix, relation_matrix[s2:e2, s1:e1])
            #     pairs.append((s1, e1, s2, e2))
        return pairs
        
    def decode_triplets(self, target_aspect_pairs, target_opinion_pairs, aspect_opinion_pairs):
        # triplets = []
        # combinations = product(target_aspect_pairs, target_opinion_pairs, aspect_opinion_pairs)
        # for elem in combinations:
        #     ts1, te1, as1, ae1, ts2, te2, os1, oe1, as2, ae2, os2, oe2 =  \
        #         elem[0][0], elem[0][1], elem[0][2], elem[0][3], elem[1][0], elem[1][1], \
        #         elem[1][2], elem[1][3], elem[2][0], elem[2][1], elem[2][2], elem[2][3]
        #     if ts1 == ts2 and te1 == te2 and as1 == as2 and ae1 == ae2 and  \
        #         os1 == os2 and oe1 == oe2:
        #             triplets.append((ts1, te1, as1, ae1, os1, oe1))
        triplets = []
        target_aspect_dict, target_opinion_dict, aspect_opinion_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        for target_aspect in target_aspect_pairs:
            target_aspect_dict[(target_aspect[0], target_aspect[1])].append((target_aspect[2], target_aspect[3]))
        for target_opinion in target_opinion_pairs:
            target_opinion_dict[(target_opinion[0], target_opinion[1])].append((target_opinion[2], target_opinion[3]))
        for aspect_opinion in aspect_opinion_pairs:
            aspect_opinion_dict[(aspect_opinion[0], aspect_opinion[1])].append((aspect_opinion[2], aspect_opinion[3]))
        
        for target, aspects in target_aspect_dict.items():
            for aspect in aspects:
                for opinion in aspect_opinion_dict[aspect]:
                    if opinion in target_opinion_dict[target]:
                        triplets.append((target[0], target[1], aspect[0], aspect[1], opinion[0], opinion[1]))
        return triplets
    
    def decode_triplets_matrix(self, entity_matrix, relation_matrix):
        targets, aspects, opinions = self.decode_entity_matrix(entity_matrix)
        target_aspect_pairs = self.decode_pairs(targets, aspects, relation_matrix)
        target_opinion_pairs = self.decode_pairs(targets, opinions, relation_matrix)
        aspect_opinion_pairs = self.decode_pairs(aspects, opinions, relation_matrix)
        triplets = self.decode_triplets(target_aspect_pairs, target_opinion_pairs, aspect_opinion_pairs)
        # triplets = []
        # target_aspect_dict, target_opinion_dict, aspect_opinion_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        # for target_aspect in target_aspect_pairs:
        #     target_aspect_dict[(target_aspect[0], target_aspect[1])].append((target_aspect[2], target_aspect[3]))
        # for target_opinion in target_opinion_pairs:
        #     target_opinion_dict[(target_opinion[0], target_opinion[1])].append((target_opinion[2], target_opinion[3]))
        # for aspect_opinion in aspect_opinion_pairs:
        #     aspect_opinion_dict[(aspect_opinion[0], aspect_opinion[1])].append((aspect_opinion[2], aspect_opinion[3]))
        # for target, aspects in target_aspect_dict.items():
        #     for aspect in aspects:
        #         for opinion in aspect_opinion_dict[aspect]:
        #             if opinion in target_opinion_dict[target]:
        #                 triplets.append((target[0], target[1], aspect[0], aspect[1], opinion[0], opinion[1]))
        return {'targets':targets, 'aspects':aspects, 'opinions':opinions,
                'target_aspect_pairs':target_aspect_pairs, 'target_opinion_pairs':target_opinion_pairs,
                'aspect_opinion_pairs':aspect_opinion_pairs, 'triplets':triplets}
    
    def encode_polarities_lsts(self, triplets):
        """
        encoder polaries
        """
        labels = []
        for t_s, t_e, a_s, a_e, o_s, o_e, polary, t_t, a_t, o_t in triplets:
            labels.append([t_s, t_e, a_s, a_e, o_s, o_e, polary])
        return labels
    
    def encode_dep(self, dep_head, dep_label, seq_len):
        """
        encoder depency relation into matrix, such as head node and edge label
        """
        head_matrix = np.zeros((seq_len, seq_len), dtype=np.int32)
        label_matrix = np.zeros((seq_len, seq_len), dtype=np.int32)
        for i in range(seq_len):
            for j, h in enumerate(dep_head[i]):
                if h > float('-inf'):
                    head_matrix[i][h] = 1
                    label_matrix[i][h] = dep_label[i][j]
        return head_matrix, label_matrix
    
    def encode_replies(self, replies):
        """
        encode reply matrix, if j is reply of i, then matrix[i][j] = 1
        """
        matrix = np.zeros((len(replies),len(replies)))
        matrix[:, 0] = 1
        thread_lens = []
        cur_thread_len = 1
        for i in replies[1:]:
            if i == 0:
                thread_lens.append(cur_thread_len)
                cur_thread_len = 1
            else:
                cur_thread_len += 1
        if cur_thread_len > 1:
            thread_lens.append(cur_thread_len)
        for i in range(len(thread_lens)):
            matrix[sum(thread_lens[:i+1]):sum(thread_lens[:i+2]), sum(thread_lens[:i+1]):sum(thread_lens[:i+2])] = 1
        return matrix
    
    def encode_replies_path(self, reply_matrix:np.ndarray):
        """
        encode reply matrix into a path format
        """
        
        for i in range(reply_matrix.shape[0]):
            parent = np.argmax(reply_matrix[i])
            if i != 0: assert parent < i
            reply_matrix[i][i] = 1
            reply_matrix[i] += reply_matrix[parent]
        reply_matrix[reply_matrix > 0] = 1
        return reply_matrix
    
    # def fast_decode(self, entity_matrix:torch.Tensor, relation_matrix:torch.Tensor):
    #     """
    #     fast version to extract pairs and triplets
    #     """
    #     #entity_matrix: (S, S, 7), relation_matrix: (S, S)
    #     seq_len = entity_matrix.size(1)
    #     targets, aspects, opinions = self.decode_entity_matrix(entity_matrix)
    #     cross_flags = np.zeros((seq_len, seq_len))
    #     entity_flags = {}
    #     entity_flags.update({idx:0 for target in targets for idx in range(target[0], target[1])})
    #     entity_flags.update({idx:1 for aspect in aspects for idx in range(aspect[0], aspect[1])})
    #     entity_flags.update({idx:2 for opinion in opinions for idx in range(opinion[0], opinion[1])})
    #     for target in targets:
    #         cross_flags[target[0]:target[1]] += 1
    #         cross_flags[:, target[0]:target[1]] += 1
    #     for aspect in aspects:
    #         cross_flags[aspect[0]:aspect[1]] += 2
    #         cross_flags[:, aspect[0]:aspect[1]] += 2
    #     for opinion in opininons:
    #         cross_flags[opinion[0]:opinion[1]] += 3
    #         cross_flags[:, opinion[0]:opinion[1]] += 3
    #     cross_flags = np.triu(cross_flag, 1)
    #     for i in range(seq_len):
    #         for j in range(seq_len):
                
class WarmupCosineDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 start_lr, intermediate_lr, end_lr,
                 last_epoch=-1, verbose='deprecated'):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs - 1
        self.start_lr = start_lr
        self.intermediate_lr = intermediate_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.start_lr + (self.intermediate_lr - self.start_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            lr = self.end_lr + (self.intermediate_lr - self.end_lr)/2 \
                *(1+math.cos((self.last_epoch-self.warmup_epochs)*math.pi/(self.total_epochs-self.warmup_epochs)))
        return [lr for _ in self.base_lrs]
    
            
        
            
        