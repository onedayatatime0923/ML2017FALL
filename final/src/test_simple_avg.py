
import jieba
jieba.dt.cache_file = 'jieva.cache.new'
import numpy as np
from util import DataManager,Vocabulary
import sys

#if len(sys.argv) != 3:
#    print('python <ans_filename>')

max_word_len=14
#word_dim_list = [50, 100, 150, 200, 250, 300, 350, 400]
WORD_DIM = int(sys.argv[3])
word_dim_list = [WORD_DIM]
test = np.zeros((5060, 6))

#with open('data/hungchi_stopword') as f:
    #stop_list = [w.replace('\n', '') for w in list(f)]
stop_list = []

for word_dim in word_dim_list: 
    print('word dim=', word_dim)
    dm =DataManager()
    voc=Vocabulary()
    dm.word_dim=word_dim
    dm.word_len=max_word_len

    voc.word2vec(sys.argv[2])
    print("reading data...",end='')
    #dm.read_test_data('data/simplify_ans','test_question','test_option')
    dm.read_test_data(sys.argv[1],'test_question','test_option')
    print("\rreading data...finish")

    print("construct data...")
    #print(dm.data['test_question'][:3])
    dm.construct_data_seq2seq('test_question',voc,'data/test_question.npy', stopwords=stop_list)
    dm.construct_data_seq2seq('test_option',voc,'data/test_option.npy',multi_seq=True, stopwords=stop_list)
    #print(dm.data['test_question'][:3])
    print("construct data...finish")
    print('test_question_seq.shape: '+str(dm.data['test_question'].shape))
    print('test_option.shape: '+str(dm.data['test_option'].shape))

    test = dm.output(dm.data['test_question'])
    test_y = np.argmax(test, axis=1)
    dm.write(test_y, sys.argv[4])
