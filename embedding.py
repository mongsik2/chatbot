import os
import sys
import json
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Data Processing
class Processing :
    '''
    데이터의 최대 token길이가 10이지만
    실제 환경에서는 얼마의 길이가 들어올지 몰라 적당한 길이 부여
    '''

    def __init__(self, max_len = 20):
        self.max_len = max_len
        self.PAD = 0

    def pad_idx_sequencing(self, q_vec):
        q_len = len(q_vec)
        diff_len = q_len - self.max_len
        if(diff_len > 0):
            q_vec = q_vec[:self.max_len]
            q_len = self.max_len
        return q_vec

class MakeDataset :
    def __init__(self):

        self.intent_data_dir = "./data/dataset/intent_data.csv"
        self.prep = Processing()

    def tokenize(self, sentence):
        '''띄어쓰기 단위로 tokenize 적용'''
        return sentence.split()

    def tokenize_dataset(self, dataset):
        '''Dataset에 tokenize 적용'''
        token_dataset = []
        for data in dataset :
            token_dataset.append(self.tokenize(data))
        return token_dataset

    def make_embed_dataset(self, ood=False):
        embed_dataset = pd.read_csv(self.intent_data_dir)
        embed_dataset = embed_dataset["question"].to_list()
        embed_dataset = self.tokenize_dataset(embed_dataset)

        return embed_dataset

# Embedding
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    '''https://radimrehurek.com/gensim/models/callbacks.html'''
    '''학습 중간에 프린트를 하기 위한 logger'''
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class MakeEmbed:
    '''https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec'''
    '''https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#online-training-resuming-training'''
    def __init__(self):
        self.model_dir = "./"
        self.vector_size = 300  # 임베딩 사이즈
        self.window_size = 3    # 몇 개의 단어를 예측을 할것인지
        self.workers = 8        # 학습 스레드 수
        self.min_count = 2      # 단어의 최소 빈도 수 (해당 수 미만은 버려진다)
        self.iter = 1000        # 1epoch당 학습 수
        self.sg = 1             # 1 : skip-gram, 0 : CROW
        self.model_file = './data/pretraining/word2vec_skipgram_{}_{}_{}'.format(self.vector_size, self.window_size, self.min_count)
        self.epoch_logger = EpochLogger()

    def word2vec_init(self) : # word2vec 초기화 및 세팅
        self.word2vec = Word2Vec(size=self.vector_size,
                                window=self.window_size,
                                workers=self.workers,
                                min_count = self.min_count,
                                compute_loss=True,
                                iter=self.iter
                                )
    
    def word2vec_build_vocab(self, dataset) : # 단어장 만들기
        self.word2vec.build_vocab(dataset)

    def word2vec_most_similar(self, query) : # 비슷한 단어 계산
        print(self.word2vec.most_similar(query))

    def word2vec_train(self, embed_dataset, epoch=0) : # 학습
        if(epoch == 0):
            epoch = self.word2vec.epochs + 1
        self.word2vec.train(
            sentences=embed_dataset,
            total_examples=self.word2vec.corpus_count,
            epochs=epoch,
            callbacks=[self.epoch_logger]
        )

        self.word2vec.save(self.model_file + '.gensim')
        self.vocab = self.word2vec.wv.index2word
        self.vocab = {word: i for i, word in enumerate(self.vocab)}

    def load_word2vec(self) :
        if not os.path.exists(self.model_file+'.gensim'):
            raise Exception("모델 로딩 실패 "+ self.model_file+'.gensim')

  
        self.word2vec = Word2Vec.load(self.model_file+'.gensim')
        self.vocab = self.word2vec.wv.index2word
        self.vocab.insert(0,"<UNK>") # vocab애 없는 토큰등장할 경우를 대비한 <UNK> 토큰을 vocab에 삽입, index 1
        self.vocab.insert(0,"<PAD>") # 길이를 맞추기 위한 padding을 위해 <PAD> 토큰을 vacab에 삽입, index 0
        self.vocab = {word: i for i, word in enumerate(self.vocab)}

    def query2idx(self, query) :
        sent_idx = []

        for word in query:
            if(self.vocab.get(word)):
                idx = self.vocab[word]
            else:
                idx = 1

            sent_idx.append(idx)

        return sent_idx

# word2index
class Preprocessing:
    '''
    데이터의 최대 token길이가 10이지만
    실제 환경에서는 얼마의 길이가 들어올지 몰라 적당한 길이 부여
    '''
    
    def __init__(self, max_len = 20):
        self.max_len = max_len
        self.PAD = 0
    
    def pad_idx_sequencing(self, q_vec):
        q_len = len(q_vec)
        diff_len = q_len - self.max_len
        if(diff_len>0):
            q_vec = q_vec[:self.max_len]
            q_len = self.max_len
        else:
            pad_vac = [0] * abs(diff_len)
            q_vec += pad_vac

        return q_vec