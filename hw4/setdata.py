
import numpy as np
import argparse
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
parser = argparse.ArgumentParser(description='Predict test for model.')
parser.add_argument('--testfile', dest='testfile',type=str,required=True)
args = parser.parse_args()
test_file=args.testfile
mytest=open(test_file,'r')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading file                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_x=[]
for line in mytest:
    test_x.append(line[line.find(',')+1:-1])
test_x=np.array(test_x)[1:]
print(test_x)
np.save('data_preprocess/test_x.npy',test_x)
print('test_x shape is: '+str(test_x.shape))
