
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster  import SpectralClustering , KMeans
import pandas as pd
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
import sys
import matplotlib.pyplot as plt
assert np and BatchNormalization
assert pd

class   Embed:
    def __init__(self):
        pass
    def set_pca(self,n):
        self.pca=PCA(n_components=n,whiten=True,svd_solver='full')
    def set_tsne(self,n):
        self.tsne=TSNE(n_components=n)
    def set_kmeans(self):
        self.kmeans=KMeans(n_clusters=2,random_state=100)
    def set_spetral(self):
        self.spetral= SpectralClustering(n_clusters=2, eigen_solver='arpack',affinity="nearest_neighbors")

class   DataManager:
    def __init__(self):
        self.data={}
    def load_image(self,path,name):
        tmp=np.load(path)
        self.data[name]=(tmp-np.mean(tmp,axis=0))/255
    def load_test(self,path,name):
        x=pd.read_csv(path)
        self.data[name]=np.array(x.iloc[:,1:])
    def reduction(self,name,Embed):
        print("reduct data by pca...",end='')
        sys.stdout.flush()
        tmp=Embed.pca.fit_transform(self.data[name])
        print("\rreduct data by pca...finish")
        self.data[name]=tmp
        '''
        print("reduct data by tsne...",end='')
        sys.stdout.flush()
        self.data[name]=Embed.tsne.fit_transform(tmp)
        print("\rreduct data by tsne...finish")
        '''
    def clust(self,data,Embed):
        #clust data by kmeans in Embed
        print("cluster data...",end='')
        sys.stdout.flush()
        res=Embed.kmeans.fit_predict(data)
        print("\rcluster data...finish")
        return res
    def autoencoder(self):
        #construct the model for autoencoder
        #return autoencoder
        input_img = Input(shape=(784,))
        #encoded = Dense(256, activation='relu')(input_img)
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)

        decoded = Dense(128, activation='relu')(encoded)
        #decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(784, activation='linear')(decoded)
        autoencoder = Model(input_img, decoded)
        return autoencoder
    def encoder(self,model):
        #construct the model for autoencoder
        #return encoder
        input_img = Input(shape=(784,))
        encoded = model.layers[1](input_img)
        encoded = model.layers[2](encoded)
        #encoded = model.layers[3](encoded)
        encoder = Model(input_img, encoded)
        return encoder
    def predict(self,data,name):
        #predict data index with the name of the data.
        res=[]
        one_c=0
        zero_c=0
        for i in self.data[name]:
            if data[i[0]]==data[i[1]]:
                res.append(1)
                one_c+=1
            else:
                res.append(0)
                zero_c+=1
        print('zero_count: %d'%zero_c)
        print('one_count: %d'%one_c)
        return np.array(res).reshape((-1,1))
    def write(self,test,path):
        #write data to path.
        idx=np.array([[j for j in range(len(test))]]).T
        test=np.hstack((idx,test))
        myoutput=pd.DataFrame(test,columns=["ID","Ans"])
        myoutput.to_csv(path,index=False)
    def plot(self,name):
        for i in range(5000):
            plt.scatter(self.data[name][i][0],self.data[name][i][1],c='r')
        for i in range(5000,10000):
            plt.scatter(self.data[name][i][0],self.data[name][i][1],c='b')
        plt.show()

        
