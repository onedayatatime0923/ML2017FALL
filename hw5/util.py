
import pandas as pd
import numpy as np
from numpy.random import permutation 
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add
from keras.layers import BatchNormalization,Dropout
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add
assert Model and BatchNormalization
assert np_utils and permutation

class DataManager:
    def __init__(self):
        self.data = {}
        self.id= {}
#  Read data from data_path
#  name       : string, name of data
#  with_label : bool, read data with
#  label or without label
    def read_data(self,path,name,with_label=True):
        input_file=pd.read_csv(path)
        if (with_label):
            userid=np.array(input_file.loc[:,'UserID']).reshape(-1,1)
            movieid=np.array(input_file.loc[:,'MovieID']).reshape(-1,1)
            rating=np.array(input_file.loc[:,'Rating']).reshape(-1,1)
            data=permutation(np.hstack((userid,movieid,rating)))
            X=data[:,0:2]
            Y=data[:,2].reshape(-1,1)
            self.data[name]=[X,Y]
        else:
            userid=np.array(input_file.loc[:,'UserID']).reshape(-1,1)
            movieid=np.array(input_file.loc[:,'MovieID']).reshape(-1,1)
            X=np.hstack((userid,movieid))
            self.data[name]=[X]
    def read_id(self,path,name):
        input_file=open(path,'r',encoding='latin-1')
        id_n=[]
        next(input_file)
        for line in input_file:
            id_n.append(int(line.split('::',1)[0]))
        maxid=max(id_n)
        self.id[name]=[id_n,maxid]
        input_file.close()
    def construct(self):
        # Headline input: meant to receive sequences of userid,
        user_input = Input(shape=(1,), name='user_input')
        movie_input = Input(shape=(1,), name='movie_input')
        # This embedding layer will encode the input sequence
        # into a sequence of dense 512-dimensional vectors.
        x = Embedding(output_dim=256,input_dim=6041,input_length=1)(user_input)
        x= Dropout(0.3)(x)
        y = Embedding(output_dim=256,input_dim=3952,input_length=1)(movie_input)
        y= Dropout(0.3)(y)
        p = Embedding(output_dim=1,input_dim=6040,input_length=1)(user_input)
        p= Dropout(0.3)(p)
        q = Embedding(output_dim=1,input_dim=3952,input_length=1)(movie_input)
        q= Dropout(0.3)(q)
        x =Flatten()(x)
        y =Flatten()(y)
        p =Flatten()(p)
        q =Flatten()(q)
        dot= Dot(1)([y,x])
        main_output= Add(name='main_output')([p,q,dot])
        model=Model(inputs=[user_input,movie_input],outputs=[main_output])
        return model
    def RMSE(self,y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
    def write(self,test,path):
        idx=np.array([[j for j in range(1,len(test)+1)]]).T
        test=np.hstack((idx,test))
        #print(test.shape)
        #print(output.shape)
        myoutput=pd.DataFrame(test,columns=["TestDataID","Rating"])
        myoutput["TestDataID"] = myoutput["TestDataID"].astype(int)
        myoutput.to_csv(path,index=False)
        
        


