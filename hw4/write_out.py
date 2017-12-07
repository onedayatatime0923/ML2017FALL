
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='Handle output.')
parser.add_argument('--test', dest='test',nargs='+',type=str,required=True)
parser.add_argument('-o', dest='output',type=str,required=True)
args = parser.parse_args()
output_file=args.output
test_y=np.load(args.test[0])
for i in range(1,len(args.test)):
    tmp=np.load(args.test[i])
    test_y+=tmp
print(test_y)
test_y=np.argmax(test_y,axis=1).reshape(-1,1)
idx=np.array([[j for j in range(len(test_y))]]).T
print(test_y.shape)
print(idx.shape)
test_y=np.hstack((idx,test_y)).astype(int)
#print(output.shape)
myoutput=pd.DataFrame(test_y,columns=["id","label"])
myoutput.to_csv(output_file,index=False)
'''
'''
