import numpy as np
import pandas as pd
import sys
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_file=sys.argv[1]
mytest=pd.read_csv(test_file)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_x=[]
for i in mytest.iloc[:,1]:
    temp=np.array(i.split()).reshape(48,48,1).astype(float)
    temp/=256
    test_x.append(temp)
test_x=np.array(test_x)
np.save('test_x.npy',test_x)
