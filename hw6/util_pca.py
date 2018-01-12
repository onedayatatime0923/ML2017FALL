
from skimage import io
import numpy as np
assert io and np

class   DataManager:
    def __init__(self):
        self.data={}
    def load_image(self,path,size,name):
        res=[]
        for i in range(size):
            res.append(io.imread(path+str(i)+'.jpg').flatten())
        self.data[name]=np.array(res).T
    def load_x_mean(self,name):
        self.x_mean=np.mean(self.data[name],axis=1).reshape(-1,1)
    def construct_pca(self,name):
        #print(X_mean.shape)
        X_mean=self.x_mean.dot(np.ones((1,self.data[name].shape[1])))
        #print(X_mean.shape)
        tmp=self.data[name].astype(float)-X_mean
        return np.linalg.svd(tmp,full_matrices=False)[0]
    def pca(self,weight,data,n=4):
        X_mean=self.x_mean.dot(np.ones((1,data.shape[1])))
        tmp=data.astype(float)-X_mean
        tmp=weight[:,:n].T.dot(tmp)
        return tmp
    def reconstruct(self,weight,data):
        res=np.zeros((len(self.x_mean),))
        for i in range(len(data)):
            res+=data[i]*weight[:,i]
        res+=self.x_mean.reshape((-1,))
        return res
    def plot(self,data,output_file):
        x=np.copy(data)
        x-=np.min(x)
        x/=np.max(x)
        x=(x*255).astype(np.uint8).reshape(600,600,3)
        io.imsave(output_file,x)
        

        
