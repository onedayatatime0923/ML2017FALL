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
feature=[(i,1) for i in range(106)]+[(0,2),(4,2)]
learning_rate=1
regulation_c=30
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
if (validate):
    train_x=np.array(mytrain_x[:-1*x1_validation_n,:])
    train_y=np.array(mytrain_y[:-1*x1_validation_n,:])
else:
    train_x=np.array(mytrain_x).astype(float)
    train_y=np.array(mytrain_y).astype(float)
#print (train_x0.shape)
#print (train_y1.shape)
#print (train_y0.shape)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       logistic model                           '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
print(train_x.shape)
train_x=set_train_x(train_x,feature)
print(train_x.shape)
(mean_x,var_x)=normalization(train_x)
np.save('logistic mean.npy',mean_x)
np.save('logistic var.npy',var_x)
train_x=np.hstack((np.ones((train_x.shape[0],1)),train_x))
print(train_x.shape)
wi=np.zeros((train_x.shape[1],1))
variance=np.zeros((train_x.shape[1],1))
#print(gradient_descent(wi,variance)[0])
#print(gradient_descent(wi,variance)[1])
i=1
check=True
while(check):
#print (str(train_x.shape)+"\r",end="")
    #(wi_new,variance)=gradient_descent(wi,variance)    
    wi_old=np.copy(wi)
    gradient_descent(wi,variance)    
    #print(wi)
    #print ("train times:"+str(i)+" "*10+str(correct_percent(compute_output(wi,train_x)-train_y)),end="\r")
    i+=1
    check=(abs(np.mean(wi-wi_old))>=10**(-8))
    #wi=wi_new
print()
print ("the final correct_percent is:"+str(correct_percent(compute_output(wi,train_x)-train_y)))
np.save('logistic model.npy',wi)
print(wi)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting validation                       '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
wi = np.load('model.npy')
'''
'''
validation_x=np.array(mytrain_x)[-1*x1_validation_n:,:]
validation_y=np.array(mytrain_y)[-1*x1_validation_n:,:]
#print(validation_y)
#print(validation_y.shape)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       validate logistic model                  '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
validation_x=set_train_x(validation_x,feature)
mean=np.ones((validation_x.shape[0],1)).dot(mean_x)
var=np.ones((validation_x.shape[0],1)).dot(var_x)
validation_x=(validation_x-mean)/var
validation_x=np.hstack((np.ones((validation_x.shape[0],1)),validation_x))
output=compute_output(wi,validation_x)
print("correct_percent is: "+str(correct_percent(validation_y-output)))
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
'''''''''''''''''''''       test logistic model                      '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
wi=np.load('logistic model.npy')
mean_x=np.load('logistic mean.npy')
var_x=np.load('logistic var.npy')
test_x=set_train_x(test_x,feature)
mean=np.ones((test_x.shape[0],1)).dot(mean_x)
var=np.ones((test_x.shape[0],1)).dot(var_x)
test_x=(test_x-mean)/var
test_x=np.hstack((np.ones((test_x.shape[0],1)),test_x))
output=compute_output(wi,test_x)
'''
'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       writing output                           '''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
output=list(output[:,0])
idx=[j for j in range(1,len(output)+1)]
output=[idx,output]
myoutput=pd.DataFrame(output,index=["id","label"]).transpose()
#print(myoutput)
myoutput.to_csv(output_file,index=False)
'''
'''
