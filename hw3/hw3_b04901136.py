
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model
import numpy as np
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
continue_file=sys.argv[1]
sgd=keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
adamax=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_x=np.load('train_x.npy')
train_y=np.load('train_y.npy')
val_x=np.load('val_x.npy')
val_y=np.load('val_y.npy')
print(train_x.shape)
print(train_y.shape)
model=load_model(continue_file)
datagen_train = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=False)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting checkpoint                       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
filepath=continue_file
checkpoint1= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint2=EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
callbacks_list = [checkpoint1,checkpoint2]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       print model                              '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
model.summary()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       fit model                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
n_batch=1024
history=model.fit_generator(datagen_train.flow(train_x,train_y, batch_size=n_batch,shuffle=True),steps_per_epoch=3*len(train_x)/n_batch, epochs=100,validation_data=(val_x,val_y),callbacks=callbacks_list, verbose=1)
