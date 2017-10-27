import pandas as pd
import numpy as np
import sys

def get_sigma(x, mean):
    mean_matrix=np.ones((x.shape[0],1)).dot(mean)
    #print(mean.shape)
    result=(x-mean_matrix).T.dot(x-mean_matrix)
    return result/x.shape[0]

def set_train_x(data,feature):
    (r,c)=data.shape
    result=np.empty((r,1))
    for j in feature:
        result=np.hstack((result,(data[:,j[0]]**j[1]).reshape(r,1)))
    return result[:,1:]
def classify(x,n0,n1,m0,m1,s):
    s_inv=np.linalg.pinv(s)
    #print(s.dot(s_inv))
    wi=(m0-m1).dot(s_inv).T
    b=(-1/2)*m0.dot(s_inv).dot(m0.T)+(1/2)*m1.dot(s_inv).dot(m1.T)+np.log(n0/n1)
    result=(x.dot(wi)+b)
    #print(result)
    return (result<0).astype(int)
def normalization(x):
    (r,c)=x.shape
    mean_trainx=np.mean(x,axis=0).reshape((1,c))
    var_trainx=np.var(x,axis=0).reshape((1,c))
    var_trainx=var_trainx**(1/2)
    mean=np.dot(np.ones((r,1)),mean_trainx)
    var=np.dot(np.ones((r,1)),var_trainx)
    x-=mean
    x/=var
    return (mean_trainx,var_trainx)
'''
def partial_D(wi):
    z=np.dot(train_x,wi)
    #print(z.shape)
    g=1/(1+np.exp(-1*z))-train_y
    regulation_wi=np.copy(wi)
    regulation_wi[0,0]=0
    result=2*train_x.T.dot(g)+(regulation_c*(regulation_wi))
    return result

def gradient_descent(wi,variance):
    #print(wi.shape)
    variance_temp=partial_D(wi)
    #print(variance_temp.shape)
    variance+=variance_temp**2
    #print (variance.shape)
    adagrad=variance**(1/2)
    #print (adagrad)
    wi-=variance_temp/learning_rate/adagrad
    #result=variance_temp/adagrad
'''
def correct_percent(x):
    r=x.shape[0]
    right=np.sum((x==0).astype(int))
    return (right/r)
def compute_output(wi,trainx):
    z=np.dot(trainx,wi)
    g=1/(1+np.exp(-1*z))
    return (g>0.5).astype(int)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
input_x_file="X_train"
input_y_file="Y_train"
'''
test_file=sys.argv[1]
output_file=sys.argv[2]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting validation n                     '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
validate=True
validate=False
x1_validation_n=400
x0_validation_n=400
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
mytrain_x=pd.read_csv(input_x_file)
mytrain_y=pd.read_csv(input_y_file)
train_x1=np.array(mytrain_x[mytrain_y.loc[:,"label"]==1]).astype(float)
train_x0=np.array(mytrain_x[mytrain_y.loc[:,"label"]==0]).astype(float)
train_y1=np.array(mytrain_y[mytrain_y.loc[:,"label"]==1]).astype(float)
train_y0=np.array(mytrain_y[mytrain_y.loc[:,"label"]==0]).astype(float)
#print (train_x0.shape)
#print (train_y1.shape)
#print (train_y0.shape)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       generative model                         '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
n_data_1=7841
n_data_0=24720
'''
mean_1=np.mean(train_x1,axis=0).reshape(1,-1)
#print(mean_1)
mean_0=np.mean(train_x0,axis=0).reshape(1,-1)
#print(mean_0)
sigma_1=get_sigma(train_x1,mean_1)
sigma_0=get_sigma(train_x0,mean_0)
#print(sigma_0)
#print(sigma_1)
sigma=(n_data_0*sigma_0+n_data_1*sigma_1)/(n_data_0+n_data_1)
#print (mean_1)
#print (mean_1.shape)
#print (sigma)
#print (sigma.shape)
#print(np.linalg.matrix_rank(sigma))
np.save("generative_mean1.npy",mean_1)
np.save("generative_mean0.npy",mean_0)
np.save("generative_sigma.npy",sigma)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting test                             '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
mytest=pd.read_csv(test_file)
#print(mytest)
test_x=np.array(mytest)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       test generative model                    '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
mean_1=np.load("generative_mean1.npy")
mean_0=np.load("generative_mean0.npy")
sigma=np.load("generative_sigma.npy")
output=classify(test_x,n_data_0,n_data_1,mean_0,mean_1,sigma)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
output=list(output[:,0])
idx=[j for j in range(1,len(output)+1)]
output=[idx,output]
myoutput=pd.DataFrame(output,index=["id","label"]).transpose()
#print(myoutput)
myoutput.to_csv(output_file,index=False)
