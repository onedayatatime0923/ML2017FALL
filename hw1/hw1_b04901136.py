import numpy as np
import csv
import sys
np.set_printoptions(precision=4,suppress=True)

def set_data(data_csv):
    data=[np.empty((data_n,day_n*hour_n))for i in range(month_n)]
    for i in range(month_n):
        for j in range(day_n):
            data[i][:,j*hour_n:(j+1)*hour_n]=data_csv[i*data_n*day_n+j*data_n:i*data_n*day_n+(j+1)*data_n,:]
    return data
def set_train_x(data,feature,hour,delim=(0,1)):
    train_x=np.empty((len(feature)*hour+1,1))
    for i in range(len(data)):
        month_x=[]
        for j in feature:
            month_x.append(data[i][j[0]]**j[1])
        month_x=np.array(month_x)[:,delim[0]:data[0].shape[1]-delim[1]]
        #print (month_x.shape)
        temp=np.ones((1,month_x.shape[1]-hour+1))
        #print(temp.shape)
        for j in range(hour):
            #print(j)
            #print (month_x[:,j+delimeter[0]:month_x.shape[1]-1*hour+j+1-delimeter[1]].shape)
            temp=np.vstack((temp,month_x[:,j:month_x.shape[1]-hour+j+1]))
        train_x=np.hstack((train_x,temp))
    return train_x[:,1:]
def normalization(x):
    (r,c)=x.shape
    mean_trainx=np.mean(x,axis=1).reshape((-1,1))
    mean_trainx[0,0]=0
    var_trainx=np.var(x,axis=1).reshape((-1,1))
    var_trainx=var_trainx**(1/2)
    for i in range(var_trainx.shape[0]):
        if (var_trainx[i,0]==0):
            var_trainx[i,0]=1
    mean=np.dot(mean_trainx,np.ones((1,x.shape[1])))
    var=np.dot(var_trainx,np.ones((1,x.shape[1])))
    return ((x-mean)/var,mean_trainx,var_trainx)
    


def partial_D(wi):
    g=np.dot(wi,train_x)-train_y
    regulation_wi=np.copy(wi.T)
    regulation_wi[0,0]=0
    result=2*(np.dot(train_x,g.T)+(regulation_c*(regulation_wi)))
    return result

def gradient_descent(wi,variance):
    variance_temp=partial_D(wi)
    variance+=variance_temp**2
    #print (variance)
    adagrad=variance**(1/2)
    #print (adagrad)
    result=(variance_temp/adagrad).T
    return (wi-result,variance)
            
def error(wi,train_x,train_y):
    return np.var((np.dot(wi,train_x)-train_y))**(1/2)


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''         SETTING READING OPTION                             '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
inputf="train.csv"
inputf="train_validation.csv"
month_n=12
data_n=18
day_n=20
day_n=18
hour_n=24
testf=sys.argv[1]#"test.csv"
test_validationf="test_validation.csv"
outputf=sys.argv[2]#"output.csv"

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''         SETTING TRAINING OPTION                            '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
train_hour=9
#train_feature=[1,6,7,8,9,10]
train_feature=[(8,1),(9,1),(8,2),(9,2),(8,3),(9,3)]
train_feature=[(i,1) for i in range(18)]+[(7,2),(8,2),(9,2),(10,2)]
#train_feature=[(i,1) for i in range(18)]+[(i,2) for i in range(18)]
regulation_c=30#int(sys.argv[1])
#print(int(sys.argv[1]))

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          FIXXED TRAINING OPTION                            '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
idx_PM25=9

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''         READING AND SETTING DATA                           '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
myfile=open(inputf,"r",encoding="big5")
data_csv=[]
for row in csv.reader(myfile):
    temp=[]
    for i in range(3,len(row)):
        if (row[i]=="NR"):
            temp.append(0)
        else:
            temp.append(float(row[i]))
    data_csv.append(temp)
data_csv=np.array(data_csv)[1:,:]
data=set_data(data_csv)
#print (data[0].shape)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''         BUILDING TRAINING DATA                             '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
train_x=set_train_x(data,train_feature,train_hour)
train_y=set_train_x(data,[(idx_PM25,1)],1,(train_hour,0))[1:,:]
(train_x,mean_trainx,var_trainx)=normalization(train_x)
np.save('mean.npy',mean_trainx)
np.save('var.npy',var_trainx)
'''

#print (train_y.shape)
#print (train_y)
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''                  TRAINING DATA                             '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''
wi=np.zeros((1,len(train_feature)*train_hour+1))

variance=np.zeros((len(train_feature)*train_hour+1,1))
#print(gradient_descent(wi,variance)[0])
#print(gradient_descent(wi,variance)[1])
for i in range(10000):
    #print (str(train_x.shape)+"\r",end="")
    (wi,variance)=gradient_descent(wi,variance)    
    print ("train times:"+str(i)+" "*10+str(error(wi,train_x,train_y))+"\r",end="")
print ("the final error is:"+str(error(wi,train_x,train_y)))

np.save('model.npy',wi)
print(wi)
'''
#print(wi)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          TESTING TRAINING DATA                             '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          SETTING OPTION FOR TESTING DATA                   '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
month_n=12
data_n=18
day_n=2
hour_n=24
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          TESTING DATA FOR VALIDATION                       '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
wi = np.load('model.npy')
    
myfile=open(test_validationf,"r",encoding="big5")
data_csv=[]
for row in csv.reader(myfile):
    temp=[]
    for i in range(3,len(row)):
        if (row[i]=="NR"):
            temp.append(0)
        else:
            temp.append(float(row[i]))
    data_csv.append(temp)
data_csv=np.array(data_csv)
data=set_data(data_csv)

test_x=set_train_x(data,train_feature,train_hour)
#print(data[0])
test_y=set_train_x(data,[(idx_PM25,1)],1,(train_hour,0))[1:,:]

mean=np.dot(mean_trainx,np.ones((1,test_x.shape[1])))
var=np.dot(var_trainx,np.ones((1,test_x.shape[1])))
test_x=(test_x-mean)/var
#print (test_x)
print ("the test error is:"+str(error(wi,test_x,test_y)))
#output=np.dot(wi,test_x)
#print (output)
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          SETTING OPTION FOR TESTING DATA                   '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
month_n=240
data_n=18
day_n=1
hour_n=9
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          TESTING DATA                                      '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
wi = np.load('model.npy')
mean_trainx= np.load('mean.npy')
var_trainx= np.load('var.npy')
myfile=open(testf,"r",encoding="big5")
data_csv=[]
for row in csv.reader(myfile):
    temp=[]
    for i in range(2,len(row)):
        if (row[i]=="NR"):
            temp.append(0)
        else:
            temp.append(float(row[i]))
    data_csv.append(temp)
data_csv=np.array(data_csv)
data=set_data(data_csv)

test_x=set_train_x(data,train_feature,train_hour,(hour_n-train_hour,0))
#print(data[0])
#print(test_x.shape)

mean=np.dot(mean_trainx,np.ones((1,test_x.shape[1])))
var=np.dot(var_trainx,np.ones((1,test_x.shape[1])))
test_x=(test_x-mean)/var
#print (test_x.shape)
output=np.dot(wi,test_x)
#print (output)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''          WRITING OUTPUT                                    '''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

myfile=open(outputf,"w",encoding="big5")
csvcursor=csv.writer(myfile)
row_write=["id","value"]
csvcursor.writerow(row_write)  
for i in range(month_n):
    row_write=["id_"+str(i),output[0,i]]
    csvcursor.writerow(row_write)  



    
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
