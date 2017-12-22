
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys 
sys.path.append("./")
from util import DataManager
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np
import sys
from keras.models import load_model
assert ModelCheckpoint and EarlyStopping and np

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
n_batch=2048
adam=keras.optimizers.Adam(clipnorm=0.0001)
adamax=keras.optimizers.Adamax(clipnorm=0.0001)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       create model                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=DataManager()
dm.read_data(sys.argv[1],"test",with_label=False)
model=load_model('./model.hdf5')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       output                                   '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_y=model.predict({'user_input':dm.data['test'][0][:,0], 'movie_input':dm.data['test'][0][:,1]}, batch_size=n_batch, verbose=1)
dm.write(test_y,sys.argv[2])






