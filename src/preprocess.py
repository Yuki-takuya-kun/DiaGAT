import os
import re
import json
import pickle
from typing import List
from collections import defaultdict, Counter
from itertools import accumulate, permutations, chain
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from src.common import WordPair, DataProcessor
from ltp import LTP
import stanza
from stanza.pipeline.core import DownloadMethod

class DataPreprocessor:
    
    stanza = None
    ltp = None
    
    def __init__(self, config):
        self.cfg = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.bert_path, cache_dir=self.cfg.huggingface_dir)
        self.wordpair = WordPair(self.cfg.lang)
        self.polarity_dict = self.cfg.polarity_dict
        self.entity_dict = self.wordpair.entity_dic
        self.lang = self.cfg.lang
        if self.lang == 'zh':
            self.ltp = LTP(cache_dir=config.huggingface_dir)
        elif self.lang == 'en':
            self.stanza_tokenize = stanza.Pipeline(lang='en', dir=self.cfg.stanza_dir, processors='tokenize', tokenize_no_ssplit=True, download_method=DownloadMethod.NONE)
            self.stanza = stanza.Pipeline(lang='en', dir=self.cfg.stanza_dir, processors='tokenize,mwt,pos,lemma,depparse', tokenize_no_ssplit=True, download_method=DownloadMethod.NONE)
        self.data_processor = DataProcessor(self.cfg, self.stanza, self.ltp)
    
    def get_save_data(self, ds_type):
        path = self.cfg['json_path'] +'_'+ self.lang
        data_set_path = os.path.join(path, f'{ds_type}.json')
        save_set_path = os.path.join(path, f'{ds_type}_processed.pickle')
        with open(data_set_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        new_dataset = []
        c = 0
        for dialogue in tqdm(dataset, desc='transfering'):
        #for dialogue in dataset:
            
            new_dialogue = self.dialogue_process(dialogue)
            c += 1
            # if c == 4 :
            #     break
            new_dataset.append(self.transform2indices(new_dialogue))
        #print(new_dataset[0].keys())
        # for key, val in new_dataset[0].items():
        #     print(key, val)
        
        with open(save_set_path, 'wb') as f:
            pickle.dump(new_dataset, f)
    
    def dialogue_process(self, dialogue):
        ori_sentences = dialogue['sentences']
        #print(ori_sentences)
        #ori_sentences = [sentence.lower() for sentence in ori_sentences]
        ori_sentences, offsets = self.adjust_ori_sentences(ori_sentences)
        ori_sentences = self.clean_tokens(ori_sentences)
        #print(ori_sentences[0].lower())
        new_sentences, dialogue['pieces2words'], dialogue['pos'], dialogue['dep_head'], dialogue['dep_label']\
            = self.syn_dep_analysis(ori_sentences)
        #print('new sentneces')
        #print(new_sentences)
        dialogue['sentences'] = new_sentences
        targets, aspects, opinions, triplets = [dialogue[key] for key in ['targets', 'aspects', 'opinions', 'triplets']]
        
        #realignment
        # print(targets)
        # print(aspects)
        # print(opinions)
        # print(triplets)
        targets, aspects, opinions, triplets = self.filter_pairs(targets, aspects, opinions, triplets)
        
        targets, aspects, opinions, triplets = self.character_realigment(ori_sentences, new_sentences, offsets, targets, aspects, opinions, triplets)
        
        dialogue['targets'], dialogue['aspects'], dialogue['opinions'], dialogue['triplets'] = targets, aspects, opinions, triplets
        
        # flattern the sentences
        news = [w for sentence in new_sentences for w in sentence]

        # confirm the index is correct
        for ts, te, t_t in targets:
            assert self.check_text(news, ts, te, t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(news, ts, te, t_t)
        for ts, te, t_t,_ in opinions:
            assert self.check_text(news, ts, te, t_t)
        
        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in dialogue['triplets']:
            self.check_text(news, t_s, t_e, t_t)
            self.check_text(news, a_s, a_e, a_t)
            self.check_text(news, o_s, o_e, o_t)
        
        return dialogue 
                        
    
    def syn_dep_analysis(self, sentences):
        if self.lang == 'zh':
            pos_dict = self.wordpair.ltp_pos_dict
            dep_dict = self.wordpair.ltp_dep_dict
            COO = 'COO'
            HEAD = 'HED'
            SELF = 'SELF'
            ST = 'ST'
            
            # tokenize and get part of speech and dependency parse all sentences
            doc = self.ltp.pipeline(sentences, tasks=['cws', 'pos', 'dep'])
            
            #clean the blank space and transfer the word into word index
            sentences_pieces = doc.cws
            sentences_pieces = [[[re.sub(r"^\s+", "", char) for char in piece.split(' ')] for piece in pieces] for pieces in sentences_pieces]
            sentences_pieces = [[self.cfg.cls] + [ " ".join(piece) for piece in pieces] for pieces in sentences_pieces]
            #sentences_pieces = [piece for pieces in sentences_pieces for piece in pieces ] # flattern the words

            # add [CLS] pos in pos
            sen_pos = doc.pos
            sen_pos = [['head'] + poses for poses in sen_pos]
            sen_pos = [pos for poses in sen_pos for pos in poses] # flattern
            
            # add [CLS] dep in dep
            sen_dep = doc.dep
            sen_dep = [{'head':[float('-inf')]+dep['head'], 'label':[None]+dep['label']} for dep in sen_dep ]
            sen_dep_label = [label for dep in sen_dep for label in dep['label']] #separate dep label and flattern
            tmp_len = [0]+[len(dep['head']) for dep in sen_dep ] # calculate the length of the heads to align the dialogue
            tmp_len = list(accumulate(tmp_len))
            sen_dep_head = [head+tmp_len[i] for i in range(len(sen_dep)) for head in sen_dep[i]['head']]
            
        elif self.lang == 'en':
            pos_dict = self.wordpair.stanza_pos_dict
            dep_dict = self.wordpair.stanza_dep_dict
            COO = 'parataxis'
            HEAD = 'root'
            SELF = 'self'
            ST = 'st'
            
            # similar to zh
            sentences_pieces, sen_pos, sen_dep_head, sen_dep_label = [], [], [], []
            doc = self.stanza(sentences)
            for sentence in doc.sentences:
                word_lst, pos, dep_head, dep_label= [self.cfg.cls],  [HEAD], [float('-inf')], [None]
                for word in sentence.words:
                    
                    word_lst.append(word.text)
                    pos.append(word.upos)
                    dep_head.append(word.head + len(sen_dep_head))
                    dep_label.append(word.deprel)
                sentences_pieces.append(word_lst)
                sen_pos += pos
                sen_dep_head += dep_head
                sen_dep_label += dep_label
        sentences_pieces = self.align_pieces(sentences, sentences_pieces)
        sentences = [[w for w in chain(pieces[1:])] for pieces in sentences_pieces]

        new_sentences, piece2words = self.align_tokens(sentences)
        
        sentences_pieces = [pieces for line in sentences_pieces for pieces in line]
        #print(len(sentences_pieces))
        word2pieces = defaultdict(list)
        for key, val in enumerate(piece2words):
            word2pieces[val].append(key)
        # transfer word to token id
        word_num, idx = 0, 0
        pieces2idx = []
        for piece in sentences_pieces:
            idxs = []
            if piece == self.cfg.cls:
                idxs.append(idx)
                idx += 1
            else:
                wt_len = len(word2pieces[word_num])
                idxs += [i+idx for i in range(wt_len)]
                idx += wt_len
                if len(idxs) == 0:
                    print('='*50)
                    print(word2pieces[word_num])
                    print('='*50)
                    print(len(word2pieces))
                    print('='*50)
                    print(piece)
                    print('='*50)
                    print(sentences)
                    print('='*50)
                    print(new_sentences)
                    print('='*50)
                    print(sentences_pieces)
                    print(word_num)
                    raise ValueError
                word_num += 1
                    
            pieces2idx.append(idxs)
        
        #record the COO nodes and transfer it to a complete graph
        elemts = set()
        tmp_coos = []
        coos = {}
        for i in range(len(sen_dep_head)):
            if sen_dep_label[i] == COO:
                head, child = sen_dep_head[i], i
                if head not in elemts and child not in elemts:
                    tmp_coos.append(set([head, child]))
                else:
                    for coo in tmp_coos:
                        if head in coo or child in coo:
                            coo.update([head, child])
                elemts.update([sen_dep_head[i], i])
        for item in tmp_coos:
            for i in range(len(item)):
                item = list(item)
                coos[item[i]] = item[:i] + item[i+1:]
        # transfer pos and dep to tokens formulation
        new_sen_pos, new_sen_dep_head, new_sen_dep_label = [], [], []
        keys = [list(coos.keys())]
        token_idx, cur_head_idx = 0, 0
        for i in range(len(sentences_pieces)):
            piece2idx = pieces2idx[i]
            # print(sen_pos)
            # print(len(sen_pos))
            # print(i)
            # print(piece2idx)
            new_sen_pos += [sen_pos[i]]*len(piece2idx)
            # if i is [CLS] token, it has no parent
            if sen_dep_label[i] is None:
                new_sen_dep_label.append([None])
                new_sen_dep_head.append([float('-inf')])
                cur_head_idx = token_idx
                token_idx += 1
            else:
                permu_coos = defaultdict(list)
                for j in range(len(piece2idx)):
                    permu_coos[piece2idx[j]] = piece2idx[:j] + piece2idx[j+1:]
                # for item in permutations(piece2idx):
                #     permu_coos[item[0]] = item[1:]
                    
                for token in piece2idx:
                    dep_head_token, dep_label_token = [], []
                    
                    #add head edge for every token
                    dep_head_token.append(cur_head_idx)
                    dep_label_token.append(HEAD)
                    
                    #add self edge for every token
                    dep_head_token.append(token)
                    dep_label_token.append(SELF)
                    
                    #add same token edge if token length of piece larger than 1
                    if len(piece2idx) > 1:
                        dep_head_token += [head for head in permu_coos[token]]
                        dep_label_token += [ST]*len(permu_coos[token])
                        
                    #add parent node
                    if sen_dep_label[i] != HEAD:
                        parent_idxs = pieces2idx[sen_dep_head[i]]
                        dep_head_token += [parent_idx for parent_idx in parent_idxs]
                        dep_label_token += [sen_dep_label[i]]*len(parent_idxs)
                    
                    #add coo node
                    if token in keys:
                        dep_head_token += [coo_head for coo_head in coos[token]]
                        dep_label_token += [COO]*len(coos[token])
                    
                    
                    new_sen_dep_head.append(dep_head_token)
                    new_sen_dep_label.append(dep_label_token)
        
        
        # transfer new_sen_pos, new_sen_dep_label to index
        new_sen_pos = [pos_dict.get(pos) for pos in new_sen_pos]
        for labels in new_sen_dep_label:
            for label in labels:
                if label is None:
                    pass
                else:
                    dep_dict[label]
        new_sen_dep_label = [[dep_dict.get(label) for label in labels] for labels in new_sen_dep_label ]  
        return new_sentences, piece2words, new_sen_pos, new_sen_dep_head, new_sen_dep_label
    
    
    def align_pieces(self, sen_tokens, sen_words):
        if self.lang == 'en':
            sen_tokens = [ [self.cfg.cls] + sentence.split(' ') for sentence in sen_tokens]

            for k in range(len(sen_tokens)):
                tokens = sen_tokens[k]
                words = sen_words[k]
                i, j, unk_token_idx = 0, 0, -1
                unk_token = ''
                while i < len(tokens):
                    while j < len(words):
                        if unk_token_idx >= 0:
                            reverse_word = words[j][::-1]
                            matchp = re.search(reverse_word, unk_token)
                            if matchp is None:
                                sen_words[k][unk_token_idx] = unk_token[::-1]
                                i, unk_token_idx = i+1, -1
                            else:
                                span = matchp.span()
                                unk_token = unk_token[:span[0]] + unk_token[span[1]:]
                                j += 1
                        elif tokens[i] == words[j]:
                            i, j = i+1, j+1
                        elif tokens[i].find(words[j]) == 0:
                            tokens[i] = tokens[i][len(words[j]):]
                            j += 1
                        elif '<UNK>' in words[j]:
                            if words[j] == '<UNK>':
                                unk_token = tokens[i][::-1]
                                unk_token_idx = j
                                j = j+1
                            elif words[j][-len('<UNK>'):] == '<UNK>':
                                prefix = words[j][:-len('<UNK>')]
                                unk_token = tokens[i][len(prefix):][::-1]
                                unk_token_idx = j
                                j = j + 1
                            else:
                                sen_words[k][i] = tokens[i]
                                i, j = i+1, j+1  
                        else:
                            print(sen_tokens)
                            print(sen_words)
                            raise ValueError('Could not match {} and {}'.format(tokens[i], words[j]))
            return sen_words

        else:
            return sen_words
        
    
    def filter_pairs(self, targets, aspects, opinions, triplets):
        flags = np.zeros(2000, dtype=np.int32)
        for x, y, z in targets: flags[x:y] += 1
        for x, y, z in aspects: flags[x:y] += 1
        for x, y, z, p in opinions: flags[x:y] += 1
        if np.all(flags <= 1): return targets, aspects, opinions, triplets
        else:
    
            target_output, target_candidates = [], []
            aspect_output, aspect_candidates = [], []
            opinion_output, opinion_candidates = [], []
            triplet_output = []
            for x, y, z in targets:
                if np.sum(flags[x:y]) == y-x:
                    target_output.append((x,y,z))
                else:
                    target_candidates.append((x,y,z))
            for x, y, z in aspects:
                if np.sum(flags[x:y]) == y-x:
                    aspect_output.append((x,y,z))
                else:
                    aspect_candidates.append((x,y,z))
            for x, y, z, p in opinions:
                if np.sum(flags[x:y]) == y-x:
                    opinion_output.append((x,y,z,p))
                else:
                    opinion_candidates.append((x,y,z,p))

            is_cross = lambda x, y: True if x[0] <= y[0] < x[1] or x[0] < y[1] <= x[1]\
                or x[0]>=y[0] and x[1] < y[1]  else False
            target_discard, aspect_discard, opinion_discard =  [], [], []
            for target in target_candidates:
                for aspect in aspect_candidates:
                    if is_cross(target[:2], aspect[:2]):
                        aspect_discard.append(aspect)
            for target in target_candidates:
                for opinion in opinion_candidates:
                    if is_cross(target[:2], opinion[:2]):
                        opinion_discard.append(opinion)
            for aspect in aspect_candidates:
                for opinion in opinion_candidates:
                    if is_cross(aspect[:2], opinion[:2]):
                        opinion_discard.append(opinion)
                        
            target_output += target_candidates
            for aspect in aspect_candidates:
                if aspect not in aspect_discard:
                    aspect_output.append(aspect)
            for opinion in opinion_candidates:
                if opinion not in opinion_discard:
                    opinion_output.append(opinion)
                    
            aspect_discard_idxs = [aspect[:2] for aspect in aspect_discard]
            opinion_discard_idxs = [opinion[:2] for opinion in opinion_discard]
            triplet_targets, triplet_aspects, triplet_opinions = [], [], []
            for t_s, t_e, a_s, a_e, o_s, o_e, p, t_t, a_t, o_t in triplets:
                triplet_targets.append((t_s, t_e, t_t))
                triplet_aspects.append((a_s, a_e, a_t))
                triplet_opinions.append((o_s, o_e, p, o_t))
            target_counter, aspect_counter, opinion_counter = \
                Counter(triplet_targets), Counter(triplet_aspects), Counter(triplet_opinions)
            for t_s, t_e, a_s, a_e, o_s, o_e, p, t_t, a_t, o_t in triplets:
                if (a_s, a_e) not in aspect_discard_idxs and (o_s, o_e) not in opinion_discard_idxs:
                    triplet_output.append((t_s, t_e, a_s, a_e, o_s, o_e, p, t_t, a_t, o_t))
                else:
                    target_counter[(t_s, t_e, t_t)] -= 1
                    aspect_counter[(a_s, a_e, a_t)] -= 1
                    opinion_counter[(o_s, o_e, p, o_t)] -= 1
            
            for key, val in target_counter.items():
                if val == 0 and key in target_output: target_output.remove(key)
            for key, val in aspect_counter.items():
                if val == 0 and key in aspect_output: aspect_output.remove(key)
            for key, val in opinion_counter.items():
                if val == 0 and key in opinion_output: opinion_output.remove(key)
            return target_output, aspect_output, opinion_output, triplet_output
               
    
    def character_realigment(self, ori_sentences, new_sentences, offsets, targets=None, aspects=None, opinions=None, triplets=None):
        """
        realigment the characters after using stanza or ltp tokenizer
        Returns:
            index bias for every character
        """

        if targets is None and aspects is None and opinions is None and triplets is None:
            raise ValueError("target, aspects, opinions and triplets are all None")
        
        ori_sen_chars = [char for sentence in ori_sentences for char in re.sub(r"\s{2,}", " ", sentence.strip()).split(' ')]
        new_sen_chars = [char for sentence in new_sentences for char in sentence]
        mapping = [[] for _ in range(len(ori_sen_chars)+1)]
        i, j = 0, 0
        while i < len(ori_sen_chars):
            while j < len(new_sen_chars):
                if ori_sen_chars[i] == new_sen_chars[j]:
                    mapping[i].append(j)
                    i, j = i+1, j+1
                else:
                    idx = ori_sen_chars[i].find(new_sen_chars[j])
                    if idx == -1 or idx > 0:
                        print(ori_sentences)
                        print(new_sentences)
                        print('ori', ori_sen_chars[i])
                        print('new', new_sen_chars[j])
                        raise ValueError
                    else:
                        ori_sen_chars[i] = ori_sen_chars[i][len(new_sen_chars[j]):]
                        mapping[i].append(j)
                        j += 1
        mapping[-1].append(j)
        offsets.append(offsets[-1])
        if targets is not None:
            new_targets = []
            for x, y, z in targets:
                if x - offsets[x] >= 0 and x - offsets[x] < len(mapping):
                    if offsets[y] == offsets[y-1]:
                        new_targets.append([mapping[x-offsets[x]][0], mapping[y-offsets[y]][0], z])
                    else:
                        new_targets.append([mapping[x-offsets[x]][0], mapping[y-offsets[y]][1], z])
                else:
                    new_targets.append([-1, -1, z])
            targets = new_targets
            # targets = [[mapping[x-offsets[x]][0], mapping[y-offsets[y]][0], z] \
            #     if x - offsets[x] >= 0 and x - offsets[x] < len(mapping) else [-1, -1, z] for x, y, z in targets]
        if aspects is not None:
            aspects = [[mapping[x-offsets[x]][0], mapping[y-offsets[y]][0], z] \
                if x - offsets[x] >= 0 and x - offsets[x] < len(mapping) else [-1, -1, z] for x, y, z in aspects]
        if opinions is not None:
            opinions = [[mapping[x-offsets[x]][0], mapping[y-offsets[y]][0], z, p] \
                if x - offsets[x] >= 0 and x - offsets[x] < len(mapping) else [-1, -1, z, p] for x, y, z, p in opinions]
        if triplets is not None:
            new_triplets = []
            for t_s, t_e, a_s, a_e, o_s, o_e, p, t_t, a_t, o_t in triplets:
                if t_s - offsets[t_s] >= 0 and t_s - offsets[t_s] < len(mapping):
                    if offsets[t_e] == offsets[t_e-1]:
                        nts, nte = mapping[t_s-offsets[t_s]][0], mapping[t_e-offsets[t_e]][0]
                    else:
                        nts, nte = mapping[t_s-offsets[t_s]][0], mapping[t_e-offsets[t_e]][1]
                else:
                    nts, nte = -1, -1
                # nts, nte = [mapping[t_s-offsets[t_s]][0], mapping[t_e-offsets[t_e]][0]] \
                #     if t_s - offsets[t_s] >= 0 and t_s - offsets[t_s] < len(mapping) else [-1, -1]
                nas, nae = [mapping[a_s-offsets[a_s]][0], mapping[a_e-offsets[a_e]][0]] \
                    if a_s - offsets[a_s] >= 0 and a_s - offsets[a_s] < len(mapping) else [-1, -1]
                nos, noe = [mapping[o_s-offsets[o_s]][0], mapping[o_e-offsets[o_e]][0]] \
                    if o_s - offsets[o_s] >= 0 and o_s - offsets[o_s] < len(mapping) else [-1, -1]
                new_triplets.append([nts, nte, nas, nae, nos, noe, p, t_t, a_t, o_t])
        
        return targets, aspects, opinions, new_triplets
        
    def transform2indices(self, dialogue):
        sentences, speakers, replies, pieces2words = [dialogue[k] for k in ['sentences', 'speakers', 'replies', 'pieces2words']]
        triplets, targets, aspects, opinions = [dialogue[k] for k in ['triplets', 'targets', 'aspects', 'opinions']]
        doc_id = dialogue['doc_id']
        
        sentence_length = list(map(lambda x: len(x) + 1, sentences)) # sentence length is the length of each sentence plus 2, which means add [CLS] and [SEP]
        token2senid = [[i]*len(sentence) for i, sentence in enumerate(sentences)] # token2id is to get the sentence id according to token sequence
        token2senid = [word for sentence in token2senid for word in sentence] # flatten it
                    
        #transfer sentences to tokens
        
        tokens = [[self.cfg.cls] + w for w in sentences]
        # print('tokens')
        # print(tokens)

        # sentence_ids of each token (new token)
        nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
        nsentence_ids = [w for line in nsentence_ids for w in line]
        
        targets = [(s + token2senid[s]+1, e + token2senid[s]+1, t) for s, e, t in targets]
        aspects = [(s + token2senid[s]+1, e + token2senid[s]+1, t) for s, e, t in aspects]
        opinions = [(s + token2senid[s] + 1, e + token2senid[s]+1, t, p) for s, e, t, p in opinions]
        opinions = list(set(opinions))
        # print('after')
        # print(targets)
        # print(aspects)
        # print(opinions)
        # print(triplets)
        
        new_triplets, pairs = [], []
        #transfer triplets index
        full_triplets = []
        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
            new_index = lambda start, end: (-1,-1) if start == -1 else (start + token2senid[start]+1, end + token2senid[start]+1)
            t_s, t_e = new_index(t_s, t_e)
            a_s, a_e = new_index(a_s, a_e)
            o_s, o_e = new_index(o_s, o_e)
            elements = (t_s, t_e, a_s, a_e, o_s, o_e, self.polarity_dict.get(polarity, self.polarity_dict.get('other')), t_t, a_t, o_t)
            full_triplets.append(elements)
            if all(w != -1 for w in [t_s, a_s, o_s]):
                    new_triplets.append(elements)
        pairs = self.get_pair(new_triplets)

        seq_len = sum(map(len, tokens))
        entities = {'targets':targets, 'aspects': aspects, 'opinions':opinions}
        entity_matrix = self.data_processor.encode_entity_matrix(targets, aspects, opinions, seq_len)
        relation_matrix = self.data_processor.encode_relation_matrix(new_triplets, seq_len)
        polarity_lists = self.data_processor.encode_polarities_lsts(new_triplets)
        
        dep_head_mat, dep_label_mat = self.data_processor.encode_dep(dialogue['dep_head'], dialogue['dep_label'], seq_len)
        reply_matrix = self.data_processor.encode_replies(replies)
        speaker_matrix = self.data_processor.encode_speakers(speakers)
        #tokens = [w + [self.cfg.sep] for w in tokens]
        input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
        input_masks = [[1] * len(w) for w in input_ids]
        input_segments = [[0] * len(w) for w in input_ids]
        
        # get token index for position embedding
        token_index = [list(range(sentence_length[0]))]
        lens = len(token_index[0])
        for i, slen in enumerate(sentence_length):
            if i == 0: continue
            if replies[i] == 0:
                distance = lens
            token_index += [list(range(distance, distance+slen))]
            distance += slen
        token_index = [w for line in token_index for w in line]
        
        thread_length = []
        s, e = 0, 1
        for i in range(len(replies)+1):
            if i == len(replies) or replies[i] == 0:
                e = i
                thread_length.append(sum(sentence_length[s:e]))
                s = i
        
        cross_mat = self.get_cross_mat(replies)
        
        data = {
            'doc_id': doc_id,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'input_segments': input_segments,
            'sentence_length': sentence_length,
            'token_index':token_index,
            'thread_length': thread_length,
            'nsentence_ids': nsentence_ids,
            'piece2words': pieces2words,
            'new_triplets': new_triplets,
            "cross_mat": cross_mat,
            #'full_triplets':full_triplets,
            'replies': replies,
            'reply_matrix':reply_matrix,
            'speakers': speakers,
            'speaker_matrix': speaker_matrix, 
            'pairs': pairs,
            'entities':entities,
            'entity_matrix': entity_matrix,
            'relation_matrix': relation_matrix,
            'polarity_lists': polarity_lists,
            'pos': dialogue['pos'],
            'dep_head': dialogue['dep_head'],
            'dep_label': dialogue['dep_label'],
            'dep_head_matrix': dep_head_mat,
            'dep_label_matrix': dep_label_mat}
        self.verify_data(data)
        return data
    
    def get_pair(self, full_triplets):
        pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            st, et, sa, ea, so, eo, p = full_triplets[i][:7]
            if st != -1 and sa != -1:
                pairs['ta'].add((st, et, sa, ea))

            if st != -1 and so != -1:
                pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))
        for key, val in pairs.items():
            pairs[key] = list(val)
        return pairs
    
    def get_cross_mat(self, replies):
        replies = [r + 1 for r in replies]
        sen_len = len(replies)
        cross_mat = np.zeros((sen_len, sen_len), dtype=int)
        threads = []
        cur_thread = [0]
        for i in range(1, sen_len+1):
            if i == sen_len or replies[i] == 1:
                threads.append(cur_thread)
                cur_thread = [i]
            else:
                cur_thread.append(i)
        
        for i, threadi in enumerate(threads):
            for ii, threadii in enumerate(threadi):
                for j, threadj in enumerate(threads):
                    for jj, threadjj in enumerate(threadj):
                        if i == j: # this indicate that two sentence belongs to one thread
                            cross_mat[threadii, threadjj] = cross_mat[threadjj, threadii] = abs(ii - jj)
                        elif i * j == 0: # this indicate that two sentence of one is head sen
                            cross_mat[threadii, threadjj] = cross_mat[threadjj, threadii] = ii + jj + 1
                        else:
                            cross_mat[threadii, threadjj] = cross_mat[threadjj, threadii] = ii + jj + 2
        return cross_mat

    
            
    
    def check_text(self, news, start_idx, end_idx, source_text):
        if len(source_text) == 0 and (start_idx != -1 or end_idx != -1):
            raise AssertionError("text length is 0 but start idx and end idx is not -1")
        tokenized_text = ''.join(news[start_idx:end_idx])
        t0 = tokenized_text.lower()
        if self.lang == 'en':
            source_tokens = [self.clean_token(token).lower() for token in source_text.split(' ')]
        elif self.lang == 'zh':
            source_tokens = [self.clean_token(token).lower() for token in source_text]
        t1 = ''.join(source_tokens).replace(' ','')
        if t0 != t1:
            print('assert error')
            print(news[start_idx:end_idx], source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1
       
            
    def align_tokens(self, sentences):
        """
        using Tokenizer to tokenize all of the sentences.
        and every token in a same word has a same piece, e.g., 'colorful man' after tokenizer 
        will be 'color ##ful man' and they will has piece [0, 0, 1]
        
        """
        
        pieces2word = [] # a list that shows each token if it is belongs to a word
        word_num = 0
        all_pieces = [] # reserve all of the sentences that after the tokenizer
        # a dialogue has many sentences, we process the sentence one by one
        for sentence in sentences:
            # split the sentence in a word, which is convenient for calculate indices
            # the chinese dataset has been seperate by space 
            #sentence = sentence.split()
            tokens = [self.tokenizer.tokenize(w) for w in sentence]
            cur_sentence = []
            # process every token one by one
            for token in tokens:
                # piece is the sub element in the token, such as "colorful" will be seperate by "color ##ful"
                for piece in token:
                    # tokens in a word holds the same piece
                    pieces2word.append(word_num)
                word_num += 1
                # dispose chinese character if seperate the word
                token = [piece if piece[:2] != '##' else piece[2:] for piece in token]
                cur_sentence += token # add tokens to the sentence
            all_pieces.append(cur_sentence)
        return all_pieces, pieces2word
    
    def clean_token(self, token):
        if len(token) == 0: return token
        token = token.replace('≥', '>=').replace('≈', '=').replace('×', 'x')
        if token in self.cfg.unkown_tokens:
            return self.cfg.unk
        else:
            return token
    
    def clean_tokens(self, sentences):
        # transfer emoji into unk token
        new_token_sens = []
        for sentence in sentences:
            new_tokens = []
            for token in re.sub(r"\s{2,}", " ", sentence.strip()).split(' '):
                new_tokens.append(self.clean_token(token))
            new_token_sens.append(new_tokens)
        # token_sens = [[char if char not in self.cfg.unkown_tokens else self.cfg.unk \
        #     for char in re.sub(r"\s{2,}", " ", sentence.strip()).split(' ')] for sentence in sentences ]
        return [' '.join(tokens) for tokens in new_token_sens]
        
    
    def adjust_ori_sentences(self, sentences:List[str])-> [List[str], List[List[int]]]:
        """
        Adjust the dataset, for example transfer i 'm into i'm, on the other hand, clean chinese 
        character while the language is english
        Returns:
            new_sentences: sentences that after adjustment
            offsets: the offset that correspond to original sentences
        """
        tokens = [re.sub(r"^\s+", " ", sentence).strip().split(" ") for sentence in sentences]
        offsets = []
        if self.lang == 'zh':
            
            return [sentence.lower() for sentence in sentences], [0 for _ in range(sum(map(len, tokens)))]
        elif self.lang == 'en':
            new_tokens = []
            offset = 0
            for sen in tokens:
                new_sen = []
                j = 0
                while j < len(sen):
                    ranges = [iter.span() for iter in re.finditer(r'[\u4e00-\u9fff]+', sen[j])]

                    if j == len(sen) - 1 and len(ranges) == 0:
                        new_sen.append(sen[j])
                        offsets.append(offset)
                        break
                    if j < len(sen)-1 and (sen[j]+sen[j+1]).lower() in self.cfg.pairs:
                        new_sen.append(sen[j]+sen[j+1])
                        offsets += [offset, offset + 1]
                        offset += 1
                        j += 2
                    elif len(ranges) > 0:
                        idxs = [0] + [idx for idx in chain(*ranges)] + [-1]
                        char = [sen[j][idxs[i]:idxs[i+1]] for i in range(0, len(idxs)-1, 2)]
                        char = ' '.join(char)
                        if len(char) > 0 :
                            new_sen.append(char)
                            offsets.append(offset)
                        else:
                            offset += 1
                            offsets.append(offset)
                        j += 1
                    else:
                        new_sen.append(sen[j])
                        offsets.append(offset)
                        j += 1
                if len(new_sen) == 0:
                    print(new_sen)
                    raise ValueError('sentence length is zero')
                new_tokens.append(new_sen)
        return [' '.join(token) for token in new_tokens], offsets
    
    def verify_data(self, data):
        # verify entity
        entites, entity_matrix = data['entities'], data['entity_matrix']
        targets, aspects, opinions = entites['targets'], entites['aspects'], entites['opinions']
        targets = [(target[0], target[1]) for target in targets]
        aspects = [(aspect[0], aspect[1]) for aspect in aspects]
        opinions = [(opinion[0], opinion[1]) for opinion in opinions]
        
        dtargets, daspects, dopinions = self.data_processor.decode_entity_matrix(entity_matrix)

        for elem in dtargets:
            if elem not in targets:
                print(targets, ' | ', dtargets)
                raise ValueError(f'{elem} not in targets')
        # for elem in targets:
        #     if elem not in dtargets:
        #         print(targets, ' | ', dtargets)
        #         raise ValueError(f'{elem} not in dtargets')
        for elem in daspects:
            if elem not in aspects:
                print(aspects, ' | ', daspects)
                raise ValueError(f'{elem} not in aspects')
        for elem in dopinions:
            if elem not in opinions:
                print(opinions, '| ', dopinions)
                raise ValueError(f'{elem} not in opinions')
        relations, triplets = data['pairs'], data['new_triplets']
        target_aspect, target_opinion, aspect_opinion = relations['ta'], relations['to'], relations['ao']
        triplets = [(triplet[0], triplet[1], triplet[2], triplet[3], triplet[4], triplet[5]) for triplet in triplets]
        relation_matrix = data['relation_matrix']
        
        dtarget_aspect, dtarget_opinion, daspect_opinion, dtriplets = \
            self.data_processor.decode_relation_matrix(dtargets, daspects, dopinions, relation_matrix, check_dis=True)
        for elem in dtarget_aspect:
            if elem not in target_aspect:
                raise ValueError(f'{elem} not in target aspect pair')
        # for elem in target_aspect:
        #     if elem not in dtarget_aspect:
        #         print(targets)
        #         print(aspects)
        #         print(target_aspect)
        #         print(dtarget_aspect)
        #         raise ValueError(f'{elem} not in dtarget aspect pair')
        for elem in dtarget_opinion:
            if elem not in target_opinion:
                print(targets)
                print(opinions)
                print(target_opinion)
                print(dtargets)
                print(dopinions)
                print(dtarget_opinion)
                raise ValueError(f'{elem} not in target opinion pair')
        for elem in daspect_opinion:
            if elem not in aspect_opinion:
                print(aspects)
                print(daspects)
                print(triplets)
                print(daspect_opinion)
                print(aspect_opinion)
                raise ValueError(f'{elem} not in aspect opinion pair')
        for elem in dtriplets:
            if elem not in triplets:
                print(targets)
                print(dtargets)
                print(triplets)
                print(dtriplets)
                raise ValueError(f'{elem} not in triplets')
               
    def process(self):
        print('='*30 + ' preparing datasets ' + '='*30)
        dataset_types = ['train', 'valid', 'test']
        for ds_type in dataset_types:
            print(f'preparing {ds_type} dataset')
            self.get_save_data(ds_type)
        print('='*30 + ' datasets has been saved ' + '='*30)
        
