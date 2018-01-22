
import sys 
sys.path.append("./")
from util import DataManager,Vocabulary
import jieba
import numpy as np
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
import argparse
assert jieba and np and ModelCheckpoint and EarlyStopping and load_model 
assert argparse

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Handle input model.')
parser.add_argument('-t','--test', dest='test',type=str,required=True)
parser.add_argument('-w','--w2v', dest='w2v',nargs='+',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
n_batch=32
max_word_len=50
word_dim=300

adam=keras.optimizers.Adam(clipnorm=0.0001)
adamax=keras.optimizers.Adamax(clipnorm=0.0001)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       test w2v embedding                       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dm =DataManager(max_word_len)
voc=Vocabulary()
output=np.zeros((5060,6))

for i in range(len(args.w2v)):
    voc.word2vec(args.w2v[i])

    print("reading data...",end='')
    dm.read_test_data('../'+args.test,'test_question','test_option')
    print("\rreading data...finish")

    print("construct data...",end='')
    dm.construct_data_nopadding('test_question',voc,None)
    dm.construct_data_nopadding('test_option',voc,None,multi_seq=True)
    print("\rconstruct data...finish")
    print('test_question_seq.shape: '+str(dm.data['test_question'].shape))
    print('test_option.shape: '+str(dm.data['test_option'].shape))
    output+=(dm.predict_seq_by_average(output='cosine'))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print(output.shape)

ans=dm.ensemble(output,mode='average')
print(ans.shape)
dm.write(ans,'../'+args.output)
'''
'''
