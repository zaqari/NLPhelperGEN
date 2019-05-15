import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import SnowballStemmer
import xml.etree.ElementTree as ET
import spacy

nonce_final='kelilili'
nonce_initial='kelsakiwi'

nlp = spacy.load('en')


class sseq2seq():

    def __init__(self, datasetpath, stemming=False):
        self.data = pd.read_csv(datasetpath, skipinitialspace=True)
        self.stemming = stemming

        self.encoder_data = None
        self.encoder_dic = {}

        self.decoder_out = None
        self.decoder_steps = None
        self.decoder_dic = {}

        self.lem = SnowballStemmer('english')
        super(seq2seq, self).__init__()

    def hash_dic_builder(self, dfk, col, enORde='en'):
        dic={}

        vocab=[]
        for loc in dfk.index:
            line = nltk.sent_tokenize(dfk[col].loc[loc])
            if self.stemming:
                line = [self.lem.stem(str(w)) for w in line]
            vocab+=line


        ct=0
        for it in set(vocab):
            if stemming:
                dic[self.lem.stem(str(it))]=ct
            else:
                dic[str(it)]=ct
            ct+=1

        if self.stemming:
            dic[self.lem.stem(nonce_final)]=ct
            ct+=1
            dic[self.lem.stem(nonce_initial)]=ct
        else:
            dic[nonce_final]=ct
            ct+=1
            dic[nonce_initial]=ct

        if enORde == 'en':
            self.encoder_dic=dic
        else:
            self.decoder_dic=dic


    def encoder_inputs(self, text_column):

        lines=[]

        dataframe = self.data

        self.hash_dic_builder(dataframe, text_column)

        for loc in dataframe.index:
            if self.stemming:
                lines.append(np.array([self.encoder_dic[self.lem.stem(str(w))] for w in nltk.sent_tokenize(dataframe[text_column].loc[loc])]).reshape(1, -1))
            else:
                lines.append(np.array([self.encoder_dic[str(w)] for w in nltk.sent_tokenize(dataframe[text_column].loc[loc])]).reshape(1, -1))

        self.encoder_data = np.array(lines)


    def decoder_outputs(self, text_column, length=15):

        lines = []

        dataframe = self.data

        self.hash_dic_builder(dataframe, text_column, 'de')

        for loc in dataframe.index:

            if self.stemming:
                a = [self.decoder_dic[self.lem.stem(str(w))] for w in nltk.sent_tokenize(dataframe[text_column].loc[loc])]
                if len(a) < length:
                    a+=[self.decoder_dic[self.lem.stem(nonce_final)] for _ in range(length-len(a))]
                    lines.append(np.array(a).reshape(1, length))
                elif len(a) > length:
                    lines.append(np.array(a[:length+1]).reshape(1, length))
                else:
                    lines.append(np.array(a).reshape(1, length))

            else:
                a=[self.decoder_dic[str(w)] for w in nltk.sent_tokenize(dataframe[text_column].loc[loc])]
                if len(a) < length:
                    a+=[self.decoder_dic[nonce_final] for _ in range(length-len(a))]
                    lines.append(np.array(a).reshape(1, length))
                elif len(a) > length:
                    lines.append(np.array(a[:length+1]).reshape(1, length))
                else:
                    lines.append(np.array(a).reshape(1, length))

        one_hot = [0.0 for _ in range(len(self.decoder_dic) + 1)]
        outputs=[]
        for line in lines:
            out=[]
            for l in line:
                a=list(one_hot)
                a[l]=1.0
                out.append(a)
            outputs.append(out)


        self.decoder_out = outputs


    def decoder_inputs(self):

        lines=[]

        for array in self.decoder_out:
            a=[]
            if self.stemming:
                a.append(self.decoder_dic[self.lem.stem(nonce_initial)])
            else:
                a.append(self.decoder_dic[nonce_initial])
                a+=[array[:-1]]
            lines.append(np.array(a).reshape(1, len(array)))

        one_hot = [0.0 for _ in range(len(self.decoder_dic) + 1)]
        outputs = []
        for line in lines:
            out = []
            for l in line:
                a = list(one_hot)
                a[l] = 1.0
                out.append(a)
            outputs.append(out)

        self.decoder_steps = np.array(lines)


    def predict_with_model(self, encoder, decoder, text, number_of_steps):
        # encoding the sentence input
        encoder_x = []
        if self.stemming:
            encoder_x = [self.encoder_dic[lem.stem(str(w))] for w in nltk.sent_tokenize(text)]
        else:
            encoder_x = [self.encoder_dic[str(w)] for w in nltk.sent_tokenize(text)]
        encoder_x = np.array(encoder_x).reshape(1, -1)
        state = encoder.predict(encoder_x)

        #Decoding the sentence sequence
        target_sequence = [0.0 for _ in range(len(self.decoder_dic)+1)]
        if self.stemming:
            target_sequence[self.decoder_dic[lem.stem(nonce_initial)]] = 1.0
        else:
            target_sequence[self.decoder_dic[nonce_initial]] = 1.0
        target_sequence = np.array(target_sequence).reshape(1, 1, len(self.decoder_dic))

        output = []
        for step in range(number_of_steps):
            y_, h, c = decoder.predict([target_sequence] + state)
            output.append(y_[0, 0, :])
            state = [h, c]
            target_sequence = keras.utils.to_categorical(y_, len(self.decoder_dic))
        return np.array(output), state[0], state[1]


    def batched_training(self, nn, epochs=10):

        for i in range(epochs):
            acc, loss = []
            print('Epoch {}/{}'.format(i+1, epochs))
            for idx in range(len(self.encoder_data)):
                enc_in = np.array(self.encoder_data[idx]).reshape(1, -1, 300)
                dec_in = np.array(self.decoder_steps[idx]).reshape(1, -1, len(self.decoder_out))
                dec_out = np.array(self.decoder_out[idx]).reshape(1, -1, len(self.decoder_out))

                nn.train_on_batch([enc_in, dec_in], dec_out, verbose=0)

            for idx in range(len(self.encoder_data)):
                enc_in = np.array(self.encoder_data[idx]).reshape(1, -1, 300)
                dec_in = np.array(self.decoder_steps[idx]).reshape(1, -1, len(self.decoder_out))
                dec_out = np.array(self.decoder_out[idx]).reshape(1, -1, len(self.decoder_out))

                los, ac = nn.evaluate([enc_in, dec_in], dec_out)
                acc.append(ac)
                loss.append(los)

            print('@acc: {} ; @los: {}'.format(sum(acc)/len(acc), sum(loss)/len(loss)))








