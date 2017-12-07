
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import load_model
import numpy as np
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Predict test for model.')
parser.add_argument('--model', dest='model',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
output_file=args.output
continue_file=args.model
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_x=np.load('data_preprocess/test_x_matrix.npy')
print(test_x.shape)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       loading model                            '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model=load_model(continue_file)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       fit model                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
n_batch=1024
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       output                                   '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_y=model.predict(test_x, batch_size=n_batch, verbose=1) 
np.save(output_file,test_y)
