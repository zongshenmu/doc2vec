#encoding=utf-8

#洗牌、构架单词字典、单词矩阵初始化、找到文档的中心单词

#read:
# knowledge.json

#write:
# docid.json

import pandas as pd
from nltk.tokenize import word_tokenize
import json
import pickle
import numpy as np

TEXT_WINDOW_SIZE = 8 #窗口大小
BATCH_SIZE = 32 * TEXT_WINDOW_SIZE #batch
EMBEDDING_SIZE = 300 #词向量大小
DOC_EMBEDDING_SIZE = 300 #文档向量大小
SHUFFLE_EVERY_X_EPOCH = 5
PV_TEST_SET_PERCENTAGE = 5
NUM_STEPS = 2001
LEARNING_RATE = 0.1
NUM_SAMPLED = 64
REPORT_EVERY_X_STEPS = 2000 #打印输出的间隔

def repeater_shuffler(l_):
    l = np.array(l_, dtype=np.int32)
    epoch = 0
    while epoch >= 0:
        if epoch % SHUFFLE_EVERY_X_EPOCH == 0:
            np.random.shuffle(l)
        for i in l:
            yield i
        epoch += 1

def build_dictionary():
    with open('./data/knowledge.json', mode='r') as dict_file:
        knowleges=json.load(dict_file)

    #统计单词的频率
    counts = count_words(knowleges)
    counts.set_value('__UNK__', 0)
    counts.set_value('__NULL__', 0)
    vocab_size = len(counts)
    #print("voc size: %s"%vocab_size)
    dictionary = {}
    for i, word in enumerate(counts.index):
        dictionary[word] = i
    reverse_dictionary = dict(zip(dictionary.values(),
                                  dictionary.keys()))
    del counts

    data = []
    doclens = []
    doc={}
    for docid, phrase in enumerate(knowleges):
        doc[docid]=phrase
        words = word_tokenize(knowleges[phrase])
        #print(words)
        for word in words:
            if word in dictionary:
                wordid = dictionary[word]
            else:
                wordid = dictionary['__UNK__']
            data.append((docid, wordid))
        # Pad with NULL values if necessary
        doclen = len(words)
        #print(doclen)
        doclens.append(doclen)
        if doclen < TEXT_WINDOW_SIZE:
            n_nulls = TEXT_WINDOW_SIZE - doclen
            data.extend([(docid, dictionary['__NULL__'])] * n_nulls)

    #字典、字典大小、data是每个单词和文档序号、doclens文本长度
    #print(data)
    #print(doclens)
    with open('./data/docid.json','w') as file:
        json.dump(doc,file)
    return dictionary, reverse_dictionary, vocab_size, data, doclens

#统计单词频率
def count_words(phrases):
    count_dict = {}
    for key,value in phrases.items():
        #word_tokenize句子分成单词
        #print(word_tokenize(value))
        words = pd.value_counts(word_tokenize(key)+word_tokenize(value))
        #print(words)
        for word, count in words.iteritems():
            try:
                count_dict[word] += count
            except:
                count_dict[word] = count
    #Series 是一个类数组的数据结构，同时带有标签（lable）或者说索引（index）
    #print(count_dict)
    return pd.Series(count_dict)

#初始化词向量矩阵
def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        # map() 会根据提供的函数对指定序列做映射。
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        # 未定义的单词为0向量
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights

#获得每个文本的中心单词组
def get_text_window_center_positions(data):
    # If TEXT_WINDOW_SIZE is even, then define text_window_center
    # as left-of-middle-pair
    doc_start_indexes = [0]
    last_docid = data[0][0]
    #d:docid
    for i, (d, _) in enumerate(data):
        if d != last_docid:
            doc_start_indexes.append(i)
            last_docid = d
    twcp = []
    #print(doc_start_indexes)
    for i in range(len(doc_start_indexes) - 1):
        twcp.extend(list(range(
            doc_start_indexes[i] + (TEXT_WINDOW_SIZE - 1) // 2,
            doc_start_indexes[i + 1] - TEXT_WINDOW_SIZE // 2
        )))
    #print(twcp)
    #twcp中每个元素是当前
    return twcp

if __name__=='__main__':
    dictionary, _, vocab_size, data, doclens = build_dictionary()
    get_text_window_center_positions(data)
    # dic={'a':1,'b':2,'c':3}
    # for i,v in dic.items():
    #     print(i,v)
