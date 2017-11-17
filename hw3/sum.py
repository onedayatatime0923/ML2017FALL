
import numpy as np
#import pandas as pd
import sys
test_y=np.load(sys.argv[1])
for i in range(2,len(sys.argv)):
    test_y+=np.load(sys.argv[i])
test_y/=len(sys.argv)-1
print (test_y)
print (test_y.shape)
np.save('test_total.npy',test_y)
