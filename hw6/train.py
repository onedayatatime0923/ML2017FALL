
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))
import sys 
sys.path.append("./")
from util import DataManager,Embed
import numpy as np
import keras
import argparse
assert np 

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='train image cluster.')
parser.add_argument('-t','--test', dest='test',type=str,required=True)
parser.add_argument('-i','--image', dest='image',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
test_file=args.test
image_file=args.image
output_file=args.output
n_batch=1024
n_epoch=100

adam=keras.optimizers.Adam(clipnorm=0.0001)
adamax=keras.optimizers.Adamax(clipnorm=0.0001)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       read and construct data                  '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm =DataManager()
em=Embed()

print("reading data...",end='')
sys.stdout.flush()
dm.load_image(image_file,'image')
dm.load_test(test_file,'test')
print("\rreading data...finish")
print('image.shape: ',dm.data['image'].shape)
print('test.shape: ',dm.data['test'].shape)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reduct data                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

em.set_pca(400)
dm.reduction('image',em)
print(dm.data['image'].shape)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       cluster data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
em.set_kmeans()
lib=dm.clust(dm.data['image'],em)
print('lib.shape: ',lib.shape)
np.save('lib.npy',lib)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_y=dm.predict(lib,'test')
dm.write(test_y,output_file)
