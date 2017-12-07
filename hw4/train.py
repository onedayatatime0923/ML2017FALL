
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
'''
import keras
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM#,GRU
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import initializers
import argparse
import numpy as np
#from gensim.models import Word2Vec
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Handle dropout.')
parser.add_argument('--dropout', dest='dropout',type=float,required=False)
args = parser.parse_args()
if (args.dropout):
    drop_out=True
    drop_rate=args.dropout
else:
    drop_out=False
    drop_rate='nodropout'
print("drop_rate: "+str(drop_rate))
n_batch=512
sequence_size=40
word_size=150
sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipvalue=0.5,clipnorm=1.)
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0, clipvalue=0.5,clipnorm=1.)
adamax=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=0.5,clipnorm=1.)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_x_file="data_preprocess/train_x_matrix.npy"
train_x_nolabel_file="data_preprocess/train_x_nolabel_matrix.npy"
train_y_file="data_preprocess/train_y.npy"
test_x_file="data_preprocess/test_x_matrix.npy"

train_x=np.load(train_x_file)
#train_x_nolabel=np.load(train_x_nolabel_file)
train_y=np.load(train_y_file)
test_x=np.load(test_x_file)
print(train_x.shape)
#print(train_x_nolabel.shape)
print(train_y.shape)
print(test_x.shape)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       create RNN model                         '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model = Sequential()
model.add(Bidirectional(LSTM(256,activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Orthogonal(gain=1.0),return_sequences=True,dropout=0.2),input_shape=(sequence_size,word_size)))
model.add(Bidirectional(LSTM(256,activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Orthogonal(gain=1.0),return_sequences=True,dropout=0.2)))
model.add(Bidirectional(LSTM(256,activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Orthogonal(gain=1.0),return_sequences=True,dropout=0.2)))
model.add(Bidirectional(LSTM(256,activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Orthogonal(gain=1.0),return_sequences=True,dropout=0.2)))
model.add(Bidirectional(LSTM(256,activation='tanh', kernel_initializer=initializers.RandomNormal(stddev=0.001),recurrent_initializer=initializers.Orthogonal(gain=1.0),dropout=0.2)))
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       create DNN model                         '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
for i in range(4):
    model.add(Dense(units=1024,activation='relu'))
    model.add(BatchNormalization())
    if (drop_out):
        model.add(Dropout(drop_rate))
model.add(Dense(units=2,activation='softmax'))
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       compile model                            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting checkpoint                       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
filepath="weights/weights-drop_out-"+str(drop_rate)+".hdf5"
checkpoint1= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2=EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       fit model                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
history=model.fit(train_x,train_y, batch_size=n_batch,shuffle=True,epochs=40,validation_split=0.05,callbacks=callbacks_list,verbose=1)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       save model                               '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
model.save('model'+str(drop_rate)+'.hdf5')
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       output                                   '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
test_y=model.predict(test_x, batch_size=n_batch, verbose=1)
np.save('test_y'+str(drop_rate)+'.npy',test_y)
'''
'''
