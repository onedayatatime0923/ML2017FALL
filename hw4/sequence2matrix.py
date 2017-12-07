from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
import numpy as np
from keras.preprocessing.text import text_to_word_sequence

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_x_file="data_preprocess/test_x.npy"
sequence_size=40
word_vec_size=150

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       construct Word2Vec                       '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
test_x=np.load(test_x_file)
model = KeyedVectors.load_word2vec_format('data_preprocess/word2vec_model.bin')
test_x_matrix=[]
for i in range(test_x.shape[0]):
    sequence=text_to_word_sequence(test_x[i])
    tmp=[]
    for j in range(sequence_size):
        if (j<(len(sequence))):
            tmp.append(model[sequence[j]])
        else:
            tmp.append(np.zeros(word_vec_size,))
    test_x_matrix.append(np.array(tmp))
    print(str(i)+'\r',end='')
test_x_matrix=np.array(test_x_matrix)
print(test_x_matrix)
print('test_x shape: ' +str(test_x_matrix.shape))
np.save('data_preprocess/test_x_matrix.npy',test_x_matrix)

