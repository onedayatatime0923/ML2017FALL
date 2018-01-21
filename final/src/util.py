
import keras.backend as K
import jieba
from gensim.models import Word2Vec
import numpy as np
import os.path
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add
from keras.layers import BatchNormalization,Dropout,GRU
from keras.layers import Bidirectional,TimeDistributed,Masking
from keras.models import Model
from scipy.spatial.distance import cosine
import pandas as pd
assert np
assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add
assert Model and BatchNormalization and Dropout and GRU and Masking

class   Vocabulary:
    def __init__(self):
        pass
    def word2vec(self,path):
        tmp= Word2Vec.load(path)
        self.W2V=tmp
class   DataManager:
    def __init__(self):
        self.data={}
        self.word_dim=300
        self.word_len=0
    def read_train_data(self,path,name):
        with open(path, 'r') as f:
            lines = list(f)
        data=[]    
        for i in range (len(lines)):
            a = lines[i].replace('\n','')
            data.append(a)
        self.data[name]= np.array(data) 
    def read_test_data(self,path,name_q,name_o):
        with open(path, 'r') as f:
            next(f)
            lines = list(f)
        questions = []
        options = []
        for i, line in enumerate(lines):
            _, question_i, options_i = line.split(',')
            questions.append(question_i.replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', ''))
            options_i = options_i.split(':')[1:]
            options_i = [opt.replace('\t','').replace('\n','').replace('A', '').replace('B', '').replace('C', '').replace('D', '') for opt in options_i]
            options.append(options_i)
        self.data[name_q]=questions
        self.data[name_o]=options
    def construct_data_LSTM(self,name,voc,outputfile,multi_seq=False):
        if(os.path.isfile(outputfile)):
            self.data[name]=np.load(outputfile)
        elif (multi_seq==False):
            vec = []
            for i in self.data[name]:
                seg = list(jieba.cut(i))
                vec_list = []
                for w in seg:
                    if (len(vec_list)>=13):break
                    elif w in voc.W2V:
                        #print(voc.W2V[w].shape)
                        vec_list.append(voc.W2V[w])
                if (len(vec_list)<13):
                    for i in range(13-len(vec_list)):
                        vec_list.append(np.zeros((self.word_dim,)))
                vec.append(vec_list)
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            np.save(outputfile,vec)
        else:
            opt_vec = []
            for opts_1Q in self.data[name]:
                opt_vec_1Q = []
                for opt in opts_1Q:
                    seg_opt = list(jieba.cut(opt))
                    opt_vec_list = []
                    for w in seg_opt:
                        if (len(opt_vec_list)>=13):break
                        elif w in voc.W2V:
                            opt_vec_list.append(voc.W2V[w])
                    if (len(opt_vec_list)<13):
                        for i in range(13-len(opt_vec_list)):
                            opt_vec_list.append(np.zeros((self.word_dim,)))
                    opt_vec_1Q.append(opt_vec_list)
                opt_vec.append(opt_vec_1Q)
            opt_vec=np.array(opt_vec)
            self.data[name]=opt_vec
            print(name,opt_vec.shape)
            np.save(outputfile,opt_vec)


    def construct_data_seq2seq(self,name,voc,outputfile,multi_seq=False, stopwords=[]):
        #if(os.path.isfile(outputfile)):
            #self.data[name]=np.load(outputfile)
        #tr4w = TextRank4Keyword()
        if (multi_seq==False):
            vec = []
            for i in self.data[name]:
                #print(i)
                #tr4w.analyze(text=i, lower=True, window=2)
                seg = list(jieba.cut(i))
                #seg = tr4w.words_all_filters[0]
                #print(seg)
                vec_list = []
                for w in seg:
                    if w in voc.W2V and w not in stopwords:
                        vec_list.append(voc.W2V[w])
                    #else:
                    #    vec_list.append([0]*self.word_dim)
                vec.append(np.array(vec_list))
            vec=np.array(vec)
            self.data[name]=vec
            #print(name,vec.shape)
            np.save(outputfile,vec)
        else:
            vec = []
            for option in self.data[name]:
                opt= []
                for i in option:
                    seg = list(jieba.cut(i))
                    vec_list = []
                    for w in seg:
                        if w in voc.W2V and w not in stopwords:
                            vec_list.append(voc.W2V[w])
                        #else:
                        #    vec_list.append([0]*self.word_dim)
                    opt.append(np.array(vec_list))
                vec.append(opt)
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            np.save(outputfile,vec)
    def wrape_encoder(self,data,voc):
        res=[]
        for i in range(len(data)):
            print('\rprocessing sequence number: '+str(i),end='')
            if(len(data[i])==0):
                res.append(np.zeros((self.word_len,self.word_dim)))
            elif(len(data[i])>self.word_len):
                data[i]=np.array(data[i])
                res.append(data[i][:self.word_len,:])
            else:
                data[i]=np.array(data[i])
                res.append(np.concatenate((data[i],np.zeros((self.word_len-len(data[i]),self.word_dim))),axis=0))
        print('\nprocess finish...')
        return np.array(res)
    def wrape_decoder(self,data,voc,decode_in=True):
        res=[]
        for i in range(len(data)):
            print('\rprocessing sequence number: '+str(i),end='')
            if (decode_in):
                if(len(data[i])==0):
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),np.zeros((self.word_len-1,self.word_dim))),axis=0))
                elif(len(data[i])>=self.word_len):
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),data[i][:self.word_len-1,:]),axis=0))
                else:
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((voc.W2V['<BOS>'].reshape((1,-1)),data[i],np.zeros((self.word_len-1-len(data[i]),self.word_dim))),axis=0))
            else:
                if(len(data[i])==0):
                    res.append(np.concatenate((np.zeros((self.word_len-1,self.word_dim)),voc.W2V['<EOS>'].reshape((1,-1))),axis=0))
                elif(len(data[i])>=self.word_len):
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((data[i][:self.word_len-1,:],voc.W2V['<EOS>'].reshape((1,-1))),axis=0))
                else:
                    data[i]=np.array(data[i])
                    res.append(np.concatenate((data[i],voc.W2V['<EOS>'].reshape((1,-1)),np.zeros((self.word_len-1-len(data[i]),self.word_dim))),axis=0))
        print('\nprocess finish...')
        return np.array(res)
    def construct_LSTM(self,unit=128):
        sequence_in= Input(shape=(self.word_len,self.word_dim), name='sequence_in')
        x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(sequence_in)
        for i in range(5):
            x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(x)
        x=TimeDistributed(Dense(unit,activation='relu'))(x)
        main_output=TimeDistributed(Dense(self.word_dim,activation='linear'),name='main_output')(x)
        model=Model(inputs=sequence_in,outputs=main_output)
        return model
    def construct_seq2seq_train(self,unit=256):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.word_dim),name='encoder_in')
        encoder_lstm=LSTM(unit, return_sequences=True,name='encoder_lstm')(encoder_inputs)
        encoder =LSTM(unit, return_state=True,name='encoder_out')
        encoder_outputs, state_h, state_c = encoder(encoder_lstm)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.word_dim),name='decoder_in')
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoder_lstm , _,_ = LSTM(unit, return_sequences=True,return_state=True,name='decoder_lstm')(decoder_inputs,initial_state=encoder_states)
        decoder= LSTM(unit, return_sequences=True, return_state=True,name='decoder_out')
        decoder_outputs, _, _ = decoder(decoder_lstm,initial_state=encoder_states)
        decoder_outputs= Dense(unit, activation='relu',name='dense_1')(decoder_outputs)
        decoder_dense = Dense(self.word_dim, activation='linear',name='dense_2')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
    def construct_seq2seq_test(self,model,unit):
        #print(model.inputs)
        #print(model.outputs)
        #print(model.layers)
        encoder_inputs = model.inputs[0] 
        encoder_lstm= model.layers[1](encoder_inputs) 
        encoder_outputs, state_h, state_c = model.layers[3](encoder_lstm)
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_inputs,encoder_states)

        decoder_inputs = Input(shape=(None, self.word_dim),name='decoder_inputs')
        decoder_lstm_input_h = Input(shape=(unit,),name='decoder_lstm_input_h')
        decoder_lstm_input_c = Input(shape=(unit,),name='decoder_lstm_input_c')
        decoder_state_input_h = Input(shape=(unit,),name='decoder_state_input_h')
        decoder_state_input_c = Input(shape=(unit,),name='decoder_state_input_c')
        decoder_lstm_inputs = [decoder_lstm_input_h, decoder_lstm_input_c]
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm ,lstm_h,lstm_c= model.layers[4](decoder_inputs, initial_state=decoder_lstm_inputs)
        decoder_lstm_states = [lstm_h, lstm_c]
        decoder_outputs, state_h, state_c = model.layers[5](decoder_lstm, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = model.layers[6](decoder_outputs)
        decoder_outputs = model.layers[7](decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_lstm_inputs+decoder_states_inputs,[decoder_outputs] + decoder_lstm_states+decoder_states)
        return [encoder_model,decoder_model]
    def decode_seq(self,data_in,model,voc):
	# Encode the input as state vectors.
        encoder_model=model[0]
        decoder_model=model[1]
        states_value =states_lstm_value= encoder_model.predict(data_in)

        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        target_seq =voc.W2V['<BOS>'].reshape((1,1,-1))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens,lstm_h,lstm_c, h, c = decoder_model.predict([target_seq] + states_lstm_value+states_value)

            # Sample a token
            #print('output_tokens.shape: ',output_tokens)
            sampled_char= voc.W2V.wv.similar_by_vector(output_tokens[0, -1, :],topn=1)[0][0]
            #print('sampled_char: ',sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<EOS>' or len(decoded_sentence) > self.word_len):
                stop_condition = True
            else:
                decoded_sentence.append(voc.W2V[sampled_char].reshape((1,-1)))
                print(sampled_char,end='')

            # Update the target sequence (of length 1).
            target_seq = voc.W2V[sampled_char].reshape((1,1,-1))

            # Update states
            states_value = [h, c]
            states_lstm_value = [lstm_h, lstm_c]
        print()
        return decoded_sentence
    def cos_distance(self,y_true, y_pred):
        def l2_normalize(x, axis):
            norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
            return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
        y_true = l2_normalize(y_true, axis=-1)
        y_pred = l2_normalize(y_pred, axis=-1)
        return K.mean(y_true * y_pred, axis=-1)
    def average(self,data):
        return np.mean(data,axis=0)
    def output(self,data):
        answer=[]
        for i in range(len(data)):
            ans=self.average(data[i])
            opt=[self.average(self.data['test_option'][i,j]) for j in range(6)]
            dist = []
            for k in range(len(opt)):
                if type(opt[k])==np.float64 or type(ans)==np.float64:
                    dist.append(0)
                else:
                    dist.append(1-cosine(ans,opt[k]))
            #dist=[1-cosine(ans,opt[k]) for k in range(len(opt)) if len()]
            answer.append(dist)
        return np.array(answer)
    def write(self,test,path):
        print('validation accu: %f '% (100*self.validate(test)))
        test=test.reshape((-1,1))
        idx=np.array([[j for j in range(1,len(test)+1)]]).T
        test=np.hstack((idx,test))
        #print(test.shape)
        #print(output.shape)
        myoutput=pd.DataFrame(test,columns=["id","ans"])
        myoutput.to_csv(path,index=False)
    def validate(self,test):
        data0=[0,0,0,0,0,0,0,0,0,0]
        data100=[0,2,5,0,0,0,0,3,0,0,
                 0,4,1,3,5,1,4,2,4,0,
                 1,2,3,4,5,1,2,3,4,5,
                 1,2,4,4,0,3,3,2,1,0,
                 2,4,0,5,4,0,3,5,2,4,
                 0,4,3,5,0,2,4,2,1,3,
                 0,2,0,2,0,1,4,5,0,0,
                 1,1,0,4,2,5,0,3,0,2]
        data5050=[3,5,0,4,3,4,5,3,0,5]
        c=0
        for i in range(len(data0)):
            if test[0+i]==data0[i]: 
                c+=1
        for i in range(len(data100)):
            if test[100+i]==data100[i]: 
                c+=1
        for i in range(len(data5050)):
            if test[5050+i]==data5050[i]: 
                c+=1
        print(c/100)
        return c/100
'''
    def write(self,test,path):
        test=test.reshape((-1,1))
        idx=np.array([[j for j in range(1,len(test)+1)]]).T
        test=np.hstack((idx,test))
        myoutput=pd.DataFrame(test,columns=["id","ans"])
        myoutput.to_csv(path,index=False)
'''
 
'''
    def dinstinct_vec(self):
        for i, question in enumerate(dm.data['test_question']):
            for x in question:
            
        #return np.array(answer)
        '''
