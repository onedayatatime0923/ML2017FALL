
import jieba
from gensim.models import Word2Vec
import numpy as np
import os.path
import keras.backend as K
from keras.utils import np_utils
from keras.layers import Input, Embedding, LSTM, Dense,Dot, Flatten, Add,Concatenate,RepeatVector
from keras.layers import BatchNormalization,Dropout,GRU 
from keras.layers import Bidirectional,TimeDistributed,Masking
from keras.models import Model
from random import randrange
from scipy.spatial.distance import cosine
from numpy.linalg import norm
import pandas as pd
assert np and randrange
assert Input and Embedding and LSTM and Dense and Dot and Flatten and Add and RepeatVector and np_utils
assert Model and BatchNormalization and Dropout and GRU and Masking and K and Concatenate and norm

class   Vocabulary:
    def __init__(self):
        pass
    def word2vec(self,path):
        tmp= Word2Vec.load(path)
        self.W2V=tmp
class   DataManager:
    def __init__(self,length):
        self.data={}
        self.word_len=length
    def set_word_dim(self,voc):
        self.word_dim=voc.W2V.vector_size
    def read_train_data(self,path,name):
        with open(path, 'r') as f:
            lines = list(f)
        data=[]    
        for i in range (len(lines)):
            a = lines[i].replace('\n','').replace('\t','')
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
            questions.append(question_i.replace('A:', '').replace('B:', '').replace('C:', '').replace('D:', '').replace('\t','').replace('\n',''))
            options_i = options_i.split(':')[1:]
            options_i = [opt.replace('\t','').replace('\n','').replace('A', '').replace('B', '').replace('C', '').replace('D', '') for opt in options_i]
            options.append(options_i)
        self.data[name_q]=questions
        self.data[name_o]=options
    def weighted(self,word,voc):
        alpha = 10
        total= len(voc.W2V.wv.vocab)
        value = alpha / np.float64(alpha+(voc.W2V.wv.vocab[word].count/total))
        return value+1 
    def construct_data_padding(self,name,voc,outputfile,multi_seq=False):
        if (outputfile!=None):
            if(os.path.isfile(outputfile)):
                self.data[name]=np.load(outputfile)
        elif (multi_seq==False):
            vec = []
            for i in self.data[name]:
                seg = list(jieba.cut(i))
                vec_list = []
                for w in seg:
                    if (len(vec_list)>=self.word_len):break
                    elif w in voc.W2V:
                        #print(voc.W2V[w].shape)
                        vec_list.append(voc.W2V[w])
                if (len(vec_list)<self.word_len):
                    for i in range(self.word_len-len(vec_list)):
                        vec_list.append(np.zeros((self.word_dim,)))
                vec.append(vec_list)
            vec=np.array(vec)
            self.data[name]=vec
            print(name,vec.shape)
            if (outputfile!=None):
                np.save(outputfile,vec)
        else:
            opt_vec = []
            for opts_1Q in self.data[name]:
                opt_vec_1Q = []
                for opt in opts_1Q:
                    seg_opt = list(jieba.cut(opt))
                    opt_vec_list = []
                    for w in seg_opt:
                        if (len(opt_vec_list)>=self.word_len):break
                        elif w in voc.W2V:
                            opt_vec_list.append(voc.W2V[w])
                    if (len(opt_vec_list)<self.word_len):
                        for i in range(self.word_len-len(opt_vec_list)):
                            opt_vec_list.append(np.zeros((self.word_dim,)))
                    opt_vec_1Q.append(opt_vec_list)
                opt_vec.append(opt_vec_1Q)
            opt_vec=np.array(opt_vec)
            self.data[name]=opt_vec
            print(name,opt_vec.shape)
            if (outputfile!=None):
                np.save(outputfile,opt_vec)
    def construct_data_nopadding(self,name,voc,outputfile,multi_seq=False):
        if (outputfile!=None):
            if(os.path.isfile(outputfile)):
                self.data[name]=np.load(outputfile)
        elif (multi_seq==False):
            vec = []
            for i in self.data[name]:
                seg =  list(jieba.cut(i))
                #print(seg)
                vec_list = []
                for w in seg:
                    if w in voc.W2V:
                        vec_list.append(voc.W2V[w]*self.weighted(w,voc))
                vec.append(np.array(vec_list))
            vec=np.array(vec)
            self.data[name]=vec
            #print(name,vec.shape)
            if (outputfile!=None):
                np.save(outputfile,vec)
        else:
            vec = []
            for option in self.data[name]:
                opt= []
                for i in option:
                    seg = list(jieba.cut(i))
                    vec_list = []
                    for w in seg:
                        if w in voc.W2V:
                            vec_list.append(voc.W2V[w]*self.weighted(w,voc))
                    opt.append(np.array(vec_list))
                vec.append(opt)
            vec=np.array(vec)
            self.data[name]=vec
            #print(name,vec.shape)
            if (outputfile!=None):
                np.save(outputfile,vec)
    def construct_test(self,name,question_name,option_name,answer_name):
        Q=[]
        O=[]
        Y=[]
        for i in range(10):
            Q.append(self.data['test_question'][i])
            O.append(self.data['test_option'][i])
        Y.extend([0,0,0,0,0,0,0,0,0,0])
        for i in range(100,180):
            Q.append(self.data['test_question'][i])
            O.append(self.data['test_option'][i])
        Y.extend([0,2,5,0,0,0,0,3,0,0,
                  0,4,1,3,5,1,4,2,4,0,
                  1,2,3,4,5,1,2,3,4,5,
                  1,2,4,4,0,3,3,2,1,0,
                  2,4,0,5,4,0,3,5,2,4,
                  0,4,3,5,0,2,4,2,1,3,
                  0,2,0,2,0,1,4,5,0,0,
                  1,1,0,4,2,5,0,3,0,2])
        for i in range(5050,5060):
            Q.append(self.data['test_question'][i])
            O.append(self.data['test_option'][i])
        Y.extend([3,5,0,4,3,4,5,3,0,5])
        self.data[name]={question_name:Q,option_name:O,answer_name:Y}
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
    def LSTM_next_seq(self,unit=128):
        sequence_in= Input(shape=(self.word_len,self.word_dim), name='sequence_in')
        x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(sequence_in)
        for i in range(5):
            x=Bidirectional(LSTM(unit,activation='tanh',return_sequences=True))(x)
        x=TimeDistributed(Dense(unit,activation='relu'))(x)
        main_output=TimeDistributed(Dense(self.word_dim,activation='linear'),name='main_output')(x)
        model=Model(inputs=sequence_in,outputs=main_output)
        return model
    def LSTM_compare_seq(self,args):
        question_i= Input(shape=(self.word_len,self.word_dim), name='question_i')
        # question input 
        x=Bidirectional(LSTM(args.unit[0],activation='tanh',return_sequences=True,dropout=args.dropout))(question_i)
        for i in range(1,len(args.unit)-1):
            x=Bidirectional(LSTM(args.unit[i],activation='tanh',return_sequences=True,dropout=args.dropout))(x)
        x=Bidirectional(LSTM(args.unit[-1],activation='tanh',dropout=args.dropout))(x)

        answer_i= Input(shape=(self.word_len,self.word_dim), name='answer_i')
        # answer input 
        y=Bidirectional(LSTM(args.unit[0],activation='tanh',return_sequences=True,dropout=args.dropout))(answer_i)
        for i in range(1,len(args.unit)-1):
            y=Bidirectional(LSTM(args.unit[i],activation='tanh',return_sequences=True,dropout=args.dropout))(y)
        y=Bidirectional(LSTM(args.unit[-1],activation='tanh',dropout=args.dropout))(y)
        # dot answer and input
        concate=Concatenate(axis=1)([x,y])
        main_output=Dense(1,activation='sigmoid',name='main_output')(concate)
        model=Model(inputs=[question_i,answer_i],outputs=main_output)
        return model
    def construct_seq2seq_train(self,unit=256):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.word_len, self.word_dim),name='encoder_in')
        encoder_lstm=Masking()(encoder_inputs)
        encoder_lstm=LSTM(unit, return_sequences=True,name='encoder_lstm')(encoder_lstm)
        _,state_h,state_c=LSTM(unit, return_state=True,name='encoder_outputs')(encoder_lstm)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states`and  as initial state.
        decoder_inputs = Input(shape=(self.word_len, self.word_dim),name='decoder_in')
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoder_lstm , _,_ = LSTM(unit, return_sequences=True,return_state=True,name='decoder_lstm')(decoder_inputs,initial_state=encoder_states)
        decoder_outputs, _, _= LSTM(unit, return_sequences=True, return_state=True,name='decoder_out')(decoder_lstm,initial_state=encoder_states)
        decoder_outputs= Dense(self.word_dim, activation='linear',name='dense_1')(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
    def construct_seq2seq_test(self,model,unit):
        #print(model.inputs)
        #print(model.outputs)
        #print(model.layers)
        encoder_inputs = model.inputs[0] 
        encoder_mask= model.layers[1](encoder_inputs) 
        encoder_lstm= model.layers[2](encoder_mask) 
        encoder_outputs, state_h, state_c = model.layers[4](encoder_lstm)
        encoder_states = [state_h, state_c]
        encoder_model = Model(encoder_inputs,encoder_states)

        decoder_inputs = Input(shape=(1, self.word_dim),name='decoder_inputs')
        #encoder_outputs= Input(shape=(unit,),name='encoder_outputs')
        #encoder_repeat= RepeatVector(1)(encoder_outputs)
        #decoder_inputs_concate=model.layers[6]([decoder_inputs,encoder_repeat])

        decoder_lstm_input_h = Input(shape=(unit,),name='decoder_lstm_input_h')
        decoder_lstm_input_c = Input(shape=(unit,),name='decoder_lstm_input_c')
        decoder_state_input_h = Input(shape=(unit,),name='decoder_state_input_h')
        decoder_state_input_c = Input(shape=(unit,),name='decoder_state_input_c')
        decoder_lstm_inputs = [decoder_lstm_input_h, decoder_lstm_input_c]
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm ,lstm_h,lstm_c= model.layers[5](decoder_inputs, initial_state=decoder_lstm_inputs)
        decoder_lstm_states = [lstm_h, lstm_c]
        decoder_outputs, state_h, state_c = model.layers[6](decoder_lstm, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = model.layers[7](decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_lstm_inputs+decoder_states_inputs,[decoder_outputs] + decoder_lstm_states+decoder_states)
        return [encoder_model,decoder_model]
    def autoencoder(self,args):
        # return model([encoder_inputs, decoder_inputs], decoder_target)
        encoder_inputs = Input(shape=(self.word_len, self.word_dim),name='encoder_in')
        encoded=Masking()(encoder_inputs)
        encoded=LSTM(args.unit[0],return_sequences=True,name='encoder_lstm1')(encoded)
        encoded=Dropout(args.dropout)(encoded)
        encoded=LSTM(args.unit[1],name='encoder_lstm2')(encoded)
        encoded=Dropout(args.dropout)(encoded)
        decoded = RepeatVector(self.word_len)(encoded)

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoded, _,_ = LSTM(args.unit[1], return_sequences=True,return_state=True,name='decoder_lstm1')(decoded)
        decoded=Dropout(args.dropout)(decoded)
        decoded, _,_ = LSTM(args.unit[0], return_sequences=True,return_state=True,name='decoder_lstm2')(decoded)
        decoded=Dropout(args.dropout)(decoded)
        decoded= Dense(self.word_dim,activation='linear')(decoded)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model(encoder_inputs, decoded)
        return model
    def encoder_decoder(self,autoencoder,args):
        #encoder: Model(encoder_inputs,[encoded]+[h_2,c_2,h_1,c_1])
        #decoder: Model([encoder_out] + decoder_lstm1_inputs+decoder_lstm2_inputs,[decoded] + decoder_lstm1_states+decoder_lstm2_states)
        encoder_inputs = Input(shape=(self.word_len, self.word_dim),name='encoder_in')
        encoded=autoencoder.layers[1](encoder_inputs) #Mask
        encoded=autoencoder.layers[2](encoded)#LSTM 64
        encoded=autoencoder.layers[3](encoded)#dropout
        encoded=autoencoder.layers[4](encoded)#LSTM 256
        encoded=autoencoder.layers[5](encoded)#dropout
        encoder_model = Model(encoder_inputs,encoded)
        # Set up the decoder, using `encoder_states`and  as initial state.
        encoder_out=Input(shape=(args.unit[1],),name='encoder_out')
        decoded = RepeatVector(1)(encoder_out)#repeat vector

        decoder_lstm1_input_h = Input(shape=(args.unit[1],),name='decoder_lstm_input_h')
        decoder_lstm1_input_c = Input(shape=(args.unit[1],),name='decoder_lstm_input_c')
        decoder_lstm2_input_h= Input(shape=(args.unit[0],),name='decoder_state_input_h')
        decoder_lstm2_input_c= Input(shape=(args.unit[0],),name='decoder_state_input_c')
        decoder_lstm1_inputs = [decoder_lstm1_input_h, decoder_lstm1_input_c]
        decoder_lstm2_inputs = [decoder_lstm2_input_h, decoder_lstm2_input_c]
        decoded,lstm1_h,lstm1_c= autoencoder.layers[7](decoded,initial_state=decoder_lstm1_inputs)
        decoder_lstm1_states = [lstm1_h, lstm1_c]
        decoded=autoencoder.layers[8](decoded)#dropout
        decoded,lstm2_h, lstm2_c= autoencoder.layers[9](decoded,initial_state=decoder_lstm2_inputs)
        decoder_lstm2_states = [lstm2_h, lstm2_c]
        decoded=autoencoder.layers[10](decoded)#dropout
        decoded= autoencoder.layers[11](decoded)
        decoder_model = Model([encoder_out] + decoder_lstm1_inputs+decoder_lstm2_inputs,[decoded] + decoder_lstm1_states+decoder_lstm2_states)

        return [encoder_model,decoder_model] 
    def decode_seq(self,data_in,model,voc,args):
	# Encode the input as state vectors.
        encoder_model=model[0]
        decoder_model=model[1]
        encoder_out= encoder_model.predict(data_in)
        lstm_states1=[np.zeros((1,args.unit[1])),np.zeros((1,args.unit[1]))]
        lstm_states2=[np.zeros((1,args.unit[0])),np.zeros((1,args.unit[0]))]

        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        #target_seq =voc.W2V['<BOS>'].reshape((1,1,-1))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens,h_1,c_1,h_2,c_2 = decoder_model.predict([encoder_out]+lstm_states1+lstm_states2)

            # Sample a token
            #print('output_tokens.shape: ',output_tokens)
            sampled_char= voc.W2V.wv.similar_by_vector(output_tokens[0, -1, :],topn=1)[0][0]
            #print('sampled_char: ',sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<EOS>' or len(decoded_sentence) > self.word_len):
                stop_condition = True
            else:
                decoded_sentence.append(sampled_char)
                print(sampled_char,' ',end='')

            # Update the target sequence (of length 1).
            #target_seq = voc.W2V[sampled_char].reshape((1,1,-1))

            # Update states
            lstm_states1=[h_1,c_1]
            lstm_states2=[h_2,c_2]
        print()
        return decoded_sentence
    def generate(self,data_list,voc,window=1,stride=1, batch_size = 1024, shuffle = True,mode='train'):
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if (mode=='train'):
                index_list = self.__get_train_order(data_list,window,stride,shuffle)
            elif (mode=='val'):
                index_list = self.__get_val_order(data_list)
            elif (mode=='test'):
                index_list = self.__get_test_order(data_list)
                print(len(index_list))
            # Generate batches
            imax = len(index_list)//batch_size
            for i in range(imax):
                # Find list of IDs
                index_list_tmp= index_list[i*batch_size:(i+1)*batch_size]
                # Generate data
                Q,O,A = self.__data_generation(data_list,voc,index_list_tmp,window)
                '''
                print('Q.shape: {}'.format(Q.shape))
                print('O.shape: {}'.format(O.shape))
                print('A.shape: {}'.format(A.shape))
                '''
                yield [Q,O],A
            # Find list of IDs
            index_list_tmp= index_list[imax*batch_size:]
            # Generate data
            Q,O,A = self.__data_generation(data_list,voc,index_list_tmp,window)
            '''
            print('Q.shape: {}'.format(Q.shape))
            print('O.shape: {}'.format(O.shape))
            print('A.shape: {}'.format(A.shape))
            '''
            yield [Q,O],A
    def __get_train_order(self,data,window,stride,shuffle):
    # Find exploration order and return shuffle index tuple list
        length=len(data)
        indexes = []
        for i in range(0,length+1-(2*window),stride):
            indexes.extend([(i,i+window,1),(i,randrange(length-(2*window)),0)]) 
        indexes = np.array(indexes)
        if (shuffle): np.random.shuffle(indexes)
        return indexes
    def __get_val_order(self,data):
    # Find exploration order and return shuffle index tuple list
        ans={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,
            100:0,101:2,102:5,103:0,104:0,105:0,106:0,107:3,108:0,109:0,
            110:0,111:4,112:1,113:3,114:5,115:1,116:4,117:2,118:4,119:0,
            120:1,121:2,122:3,123:4,124:5,125:1,126:2,127:3,128:4,129:5,
            130:1,131:2,132:4,133:4,134:0,135:3,136:3,137:2,138:1,139:0,
            140:2,141:4,142:0,143:5,144:4,145:0,146:3,147:5,148:2,149:4,
            150:0,151:4,152:3,153:5,154:0,155:2,156:4,157:2,158:1,159:3,
            160:0,161:2,162:0,163:2,164:0,165:1,166:4,167:5,168:0,169:0,
            170:1,171:1,172:0,173:4,174:2,175:5,176:0,177:3,178:0,179:2,
            5050:3,5051:5,5052:0,5053:4,5054:3,5055:4,5056:5,5057:3,5058:0,5059:5}
        indexes = []
        for i in ans:
            for j in range(1,7):
                if j-1==ans[i]:
                    indexes.append((i*7,i*7+j,0))
                else:
                    indexes.append((i*7,i*7+j,1))
        indexes = np.array(indexes)
        return indexes
    def __get_test_order(self,data):
    # Find exploration order and return shuffle index tuple list
        length=len(data)
        indexes = []
        for i in range(0,length,7):
            indexes.extend([(i,i+j,0) for j in range(1,7)])
        indexes = np.array(indexes)
        return indexes
    def __data_generation(self,data_list,voc,id_list,window):
        Q =[] 
        O =[] 
        A =[]
        # Generate data
        for i in id_list:
            #print(i[0],i[1])
            ques=''
            opt=''
            for j in range(window):
                ques+=data_list[i[0]+j]
                opt+=data_list[i[1]+j]
            Q.append(self.translate(ques,voc))
            O.append(self.translate(opt,voc))
            A.append(i[2])
        return np.array(Q),np.array(O),np.array(A)
    def translate(self,data,voc):
        seg = list(jieba.cut(data))
        vec_list = []
        for w in seg:
            if (len(vec_list)<self.word_len and w in voc.W2V):
                #print(voc.W2V[w].shape)
                vec_list.append(voc.W2V[w])
        if (len(vec_list)<self.word_len):
            for i in range(self.word_len-len(vec_list)):
                vec_list.append(np.zeros((self.word_dim,)))
        vec_list=np.array(vec_list)
        return vec_list
    def predict_seq_by_embedding(self,model,questions,options,batch_size=8192):
        answer=[]
        for i in range(len(questions)):
            tmp=model.predict(questions[i].reshape((1,14,300)))[0]
            #print(tmp.shape)
            ans=np.array(tmp).reshape((1,-1))
            opt=[np.array(model.predict(options[i][j].reshape((1,14,300)))[0]).reshape((1,-1)) for j in range(6)]
            dist=[-norm(ans-opt[k]) for k in range(len(opt))]
            answer.append(dist)
            print('\rdecoding: '+str(i),end='')
        return np.argmax(np.array(answer),axis=1)
    def predict_seq_by_average(self,output):
        answer=[]
        for i in range(len(self.data['test_question'])):
            ans=self.__average(self.data['test_question'][i])
            opt=[self.__average(self.data['test_option'][i,j]) for j in range(6)]
            dist=[]
            for k in range(len(opt)):
                if type(opt[k])==np.float64 or type(ans)==np.float64 :
                    dist.append(float('-inf'))
                else:
                    dist.append(-cosine(ans,opt[k]))
            answer.append(dist)
        if output=='answer':
            return np.argmax(np.array(answer),axis=1)
        elif output=='cosine':
            return np.array(answer)
        else : raise Exception('You have to choose mode of prediction.')
    def __average(self,data):
        return np.mean(data,axis=0)
    def search_seq(self,data):
        data_name=['train1','train2','train3','train4','train5']
        max_prob=0
        for i in data_name:
            for j in range(len(self.data[i])):
                if (1-cosine(self.data[i][j],data)>max_prob):
                    max_prob=1-cosine(self.data[i][j],data)
                    name=i
                    index=j
        return name,index
    def ensemble(self,data,mode='vote'):
        if mode=='vote':
            ans=[]
            for i in data:
                ans.append(np.argmax(np.bincount(i)))
            return np.array(ans)
        elif mode=='average':
            return np.argmax(data,axis=1)
        else : raise Exception('You have to choose mode of ensemble.')
    def write(self,test,path):
        print('validation accu: %f '% (100*self.validate(test)))
        test=test.reshape((-1,1))
        idx=np.array([[j for j in range(1,len(test)+1)]]).T
        test=np.hstack((idx,test))
        #print(test.shape)
        #print(output.shape)
        myoutput=pd.DataFrame(test,columns=["id","ans"])
        myoutput.to_csv(path,index=False)
    def validate(self,test,answer_path='answer.csv'):
        answer=[]
        with open(answer_path, 'r') as answer_file:
            next(answer_file)
            for line in list(answer_file):
                ans=line.strip('\n').split(',')
                answer.append([int(ans[0])-1,int(ans[1])])
        c=0
        for i in range(len(answer)):
            if answer[i][1]==test[answer[i][0]]:
                c+=1
        return c/len(answer)

'''
    def average_data(self,name):
        res=[]
        for i in self.data[name]:
            res.append(self.average(i))
        self.data[name]=np.array(res)
    def autoencoder(self):
        # return model([encoder_inputs, decoder_inputs], decoder_target)
        encoder_inputs = Input(shape=(self.word_len, self.word_dim),name='encoder_in')
        encoded=Masking()(encoder_inputs)
        encoded,h_1,c_1=LSTM(256, return_sequences=True,return_state=True,name='encoder_lstm1')(encoded)
        encoded,h_2,c_2=LSTM(64,return_state=True,name='encoder_lstm2')(encoded)
        decoded = RepeatVector(self.word_len)(encoded)
        # Set up the decoder, using `encoder_states`and  as initial state.
        decoder_states_1=[h_1,c_1]
        decoder_states_2=[h_2,c_2]

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the 
        # return states in the training model, but we will use them in inference.
        decoded, _,_ = LSTM(64, return_sequences=True,return_state=True,name='decoder_lstm1')(decoded,initial_state=decoder_states_2)
        decoded, _,_ = LSTM(256, return_sequences=True,return_state=True,name='decoder_lstm2')(decoded,initial_state=decoder_states_1)
        decoded= Dense(self.word_dim,activation='linear')(decoded)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model(encoder_inputs, decoded)
        return model
    def encoder_decoder(self,autoencoder):
        #encoder: Model(encoder_inputs,[encoded]+[h_2,c_2,h_1,c_1])
        #decoder: Model([encoder_out] + decoder_lstm1_inputs+decoder_lstm2_inputs,[decoded] + decoder_lstm1_states+decoder_lstm2_states)
        print(autoencoder.layers)
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.word_len, self.word_dim),name='encoder_in')
        encoded=autoencoder.layers[1](encoder_inputs) #Mask
        encoded,h_1,c_1=autoencoder.layers[2](encoded)#LSTM 64
        encoded,h_2,c_2=autoencoder.layers[3](encoded)#LSTM 16
        encoder_model = Model(encoder_inputs,[encoded]+[h_2,c_2,h_1,c_1])
        # Set up the decoder, using `encoder_states`and  as initial state.
        encoder_out=Input(shape=(64,),name='encoder_out')
        decoded = RepeatVector(1)(encoder_out)#repeat vector

        decoder_lstm1_input_h = Input(shape=(64,),name='decoder_lstm_input_h')
        decoder_lstm1_input_c = Input(shape=(64,),name='decoder_lstm_input_c')
        decoder_lstm2_input_c= Input(shape=(256,),name='decoder_state_input_h')
        decoder_lstm2_input_h= Input(shape=(256,),name='decoder_state_input_c')
        decoder_lstm1_inputs = [decoder_lstm1_input_h, decoder_lstm1_input_c]
        decoder_lstm2_inputs = [decoder_lstm2_input_h, decoder_lstm2_input_c]
        decoded,lstm1_h,lstm1_c= autoencoder.layers[5](decoded, initial_state=decoder_lstm1_inputs)
        decoder_lstm1_states = [lstm1_h, lstm1_c]
        decoded,lstm2_h, lstm2_c= autoencoder.layers[6](decoded, initial_state=decoder_lstm2_inputs)
        decoder_lstm2_states = [lstm2_h, lstm2_c]
        decoded= autoencoder.layers[7](decoded)
        decoder_model = Model([encoder_out] + decoder_lstm1_inputs+decoder_lstm2_inputs,[decoded] + decoder_lstm1_states+decoder_lstm2_states)

        return [encoder_model,decoder_model] 
    def decode_seq(self,data_in,model,voc):
	# Encode the input as state vectors.
        encoder_model=model[0]
        decoder_model=model[1]
        encoder_out,h_1,c_1,h_2,c_2,= encoder_model.predict(data_in)
        lstm_states1=[h_1,c_1]
        lstm_states2=[h_2,c_2]

        # Generate empty target sequence of length 1.
        # Populate the first character of target sequence with the start character.
        #target_seq =voc.W2V['<BOS>'].reshape((1,1,-1))

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens,h_1,c_1,h_2,c_2 = decoder_model.predict([encoder_out]+lstm_states1+lstm_states2)

            # Sample a token
            #print('output_tokens.shape: ',output_tokens)
            sampled_char= voc.W2V.wv.similar_by_vector(output_tokens[0, -1, :],topn=1)[0][0]
            #print('sampled_char: ',sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '<EOS>' or len(decoded_sentence) > self.word_len):
                stop_condition = True
            else:
                decoded_sentence.append(sampled_char)
                print(sampled_char,' ',end='')

            # Update the target sequence (of length 1).
            #target_seq = voc.W2V[sampled_char].reshape((1,1,-1))

            # Update states
            lstm_states1=[h_1,c_1]
            lstm_states2=[h_2,c_2]
        print()
        return decoded_sentence
'''
