import pandas as pd
import sys
import numpy as np

answers = []
for ans in sys.argv[1:-1]:
    answers.append(np.array(pd.read_csv(ans)['ans']))

#print(answers[0].shape)
#print(len(answers))
question_num = answers[0].shape[0]
answers = np.array(answers).T

ensemble = []
with open(sys.argv[-1], 'w') as f:
    f.write('id,ans\n')
    for i in range(question_num):
        count = np.bincount(answers[i])
        f.write('{},{}\n'.format(i+1, np.argmax(count)))

