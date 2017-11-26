from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import pdb
import pickle
import os


class Sentence_Embedding(object):
    """docstring for Sentence_Embedding"""
    def __init__(self, freq_file, wvec_file):
        # self.freqs = pd.read_csv(freq_file, sep=",")
        self.wvec_file = wvec_file
        self.w_vec_dict = dict()
        self.w_prob_dict = dict()
        self.create_w_prob_dict()
        self.get_w_vec_dict()
        # self.prune_word_vec()
        self.a = 1e-3
        # self.wvec_file = pd.read_csv(wvec_file, sep=" ", header=None, usecols=[0])

    def embed_sent(self, sentences):
        tot_sent = len(sentences)
        self.sent_vec = np.zeros((tot_sent, 300))
        for ind, sent in enumerate(sentences):
            words = sent.lower().split()
            for w in words:
                w_vec = self.get_w_vec(w)
                self.sent_vec[ind] += self.a/(self.a+self.get_w_prob(w)) * w_vec
            self.sent_vec[ind] /= len(words)
        # self.sent_vec is X
        U1, Sig1, V1 = np.linalg.svd(self.sent_vec, full_matrices=False)
        u = U1[:, 0]
        u = u[:, None]
        self.sent_vec = self.sent_vec - np.dot(u, np.dot(u.T, self.sent_vec))
        return self.sent_vec

    def get_w_vec(self, w):
        try:
            return self.w_vec_dict[w.lower()]
        except KeyError:
            return self.w_vec_dict['<unk>']

    def get_w_prob(self, w):
        try:
            return self.w_prob_dict[w.lower()]
        except KeyError:
            return self.w_prob_dict['<unk>']

    def create_w_prob_dict(self):
        # self.w_prob_dict = dict(zip(list(self.freqs['Word']), list(self.freqs['Probability'])))''
        inp_dict = open('./data/w_prob_dict.pkl', 'rb')
        self.w_prob_dict = pickle.load(inp_dict)
        inp_dict.close()
        self.w_prob_dict['<unk>'] = 0
        return

    def get_w_vec_dict(self):
        inp_dict = open('./data/w_vec_dict.pkl', 'rb')
        self.w_vec_dict = pickle.load(inp_dict)
        inp_dict.close()
        self.w_vec_dict['<unk>'] = np.zeros(300)
        return

    def prune_word_vec(self):
        # words = dict(zip(list(self.freqs['Word']), [0]*len(self.freqs['Word'])))
        # pdb.set_trace()
        with open(self.wvec_file, 'r') as f:
            line = " "
            while not(line == ""):
                line = f.readline()
                tmp_line = line.split(" ")
                if tmp_line[0] in self.w_prob_dict:
                    self.w_vec_dict[tmp_line[0]] = np.array(tmp_line[1:-1], dtype=np.float)
        out_f = open('./data/w_vec_dict.pkl', 'wb')
        pickle.dump(self.w_vec_dict, out_f)
        out_f.close()
        return


def get_word_prob_from_corpus(fname):
    out_dict = dict()
    with open(fname, 'r') as f:
        for line in f:
            tmp_line = line.split()
            for w in tmp_line:
                w = w.lower()
                if w in out_dict:
                    out_dict[w] += 1
                    if out_dict[w] == 100000:
                        print(w)
                else:
                    out_dict[w] = 1
    tot = np.sum(list(out_dict.values()))
    # pdb.set_trace()
    out_dict = dict(zip(out_dict.keys(), np.array(list(out_dict.values()))/tot))
    out_f = open('./data/w_prob_dict.pkl', 'wb')
    pickle.dump(out_dict, out_f)
    out_f.close()
    return out_dict


def get_w_prob_from_csv(fname):
    freqs = pd.read_csv(fname)
    w_prob_dict = dict(zip(list(freqs['Word']), list(freqs['Probability'])))
    # pdb.set_trace()
    out_f = open('./data/w_prob_dict.pkl', 'wb')
    pickle.dump(w_prob_dict, out_f)
    out_f.close()
    return


def scrape_sentences():
    # doing only the hub part
    # top_dir = './data/vcc2018_training/Transcriptions/'
    top_dir = './dataset/vcc2018/wav/Training Set/Transcriptions/'
    # sentences = dict()
    sub_dir = [top_dir + 'HUB/', top_dir + 'SPOKE/']
    sentences = []
    for t in sub_dir:
        for fname in os.listdir(t)[:]:
            with open(t + fname, 'r') as f:
                sent = f.readline().split()
                sent = ' '.join(sent)
                sentences.append(sent)
            # sentences[fname] = sent
    return sentences


if __name__ == '__main__':
    freq_file = './data/word_freqencies.csv'
    wvec_file = './data/wiki_new.csv'
    sentences = scrape_sentences()
    # pdb.set_trace()
    # print(sentences[:3])
    # sentences.append('How are you')
    # sentences.append('I am fine')
    # sentences.append('I am good')
    s = Sentence_Embedding(freq_file, wvec_file)
    sent_vec = s.embed_sent(sentences)
    # key_list = sentences.keys()
    sent_emb = dict()
    for i in range(sent_vec.shape[0]):
        if i < 81:
            sent_emb[str(i+10001) + '.wav'] = sent_vec[i]
        else:
            sent_emb[str(i - 81 + 20001) + '.wav'] = sent_vec[i]
    with open('./data/sent_emb.pkl', 'wb') as f:
        pickle.dump(sent_emb, f)
