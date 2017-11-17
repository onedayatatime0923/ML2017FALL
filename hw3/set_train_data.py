
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from keras.utils import np_utils
import pandas as pd
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_file="train.csv"
mytrain=pd.read_csv(train_file)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
#print(len(mytrain))
train_x=[]
val_x=[]
for i in range(len(mytrain)):
    if(i%20==7):
        val_x.append(np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)/256)
    else:
        temp=np.array(mytrain.iloc[i,1].split()).reshape(48,48,1).astype(float)
        temp/=256
        train_x.append(temp)
        train_x.append(np.flip(temp,axis=1))
train_x=np.array(train_x)
val_x=np.array(val_x)
#print(train_x.shape)
#print(val_x.shape)
np.save('train_x.npy',train_x)
np.save('val_x.npy',val_x)
'''
test_x=[]
for i in mytest.iloc[:,1]:
    temp=np.array(i.split()).reshape(48,48,1).astype(float)
    temp/=256
    test_x.append(temp)
test_x=np.array(test_x)
#print(train_x)
#print(train_x.shape)
np.save('test_x.npy',test_x)
'''
train_y=[]
val_y=[]
for i in range(len(mytrain)):
    if(i%20==7):
        val_y.append(int(mytrain.iloc[i,0]))
    else:
        train_y.append(int(mytrain.iloc[i,0]))
        train_y.append(int(mytrain.iloc[i,0]))
train_y=np.array(train_y)
train_y=np_utils.to_categorical(train_y, 7)
val_y=np.array(val_y)
val_y=np_utils.to_categorical(val_y, 7)
np.save('train_y.npy',train_y)
np.save('val_y.npy',val_y)
#print(train_y.shape)
#print(val_y.shape)
