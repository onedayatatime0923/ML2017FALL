
import numpy as np
import pandas as pd
import sys
output_file=sys.argv[2]
output = np.load(sys.argv[1])
output=np.argmax(output,axis=1)
idx=[j for j in range(len(output))]
output=list(output)
output=[idx,output]
output=np.array(output).T
#print(output.shape)
myoutput=pd.DataFrame(output,columns=["id","label"])
myoutput.to_csv(output_file,index=False)
'''
'''
