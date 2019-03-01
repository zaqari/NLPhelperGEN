import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import SnowballStemmer
import spacy

nonce='kelililili'

nlp = spacy.load('en')

class yelp():

    def __init__(self, stemming=False, binary=True, padding=False):
        self.reviews = None
        self.labels = None
        self.y = None
        self.data = None
        self.business = None
        self.padding = padding
        self.maxlen = None

        self.n_classes = None
        if binary:
            self.n_classes = 2
        else:
            self.n_classes = 5
        self.binary = binary

        self.lem = SnowballStemmer('english')
        self.stemming = stemming
        self.word2id = {}
        self.yelp_data()
        self.word2id_builder()
        super(yelp, self).__init__()

    def yelp_data(self):
        lines = []
        labels = []
        business_id = []
        df = pd.read_csv('NLPhelperGEN/datasets/yelp.csv', skipinitialspace=True)
        for loc in df.index:
            text = df['text'].loc[loc]
            punctionless = [char for char in text if char not in string.punctuation]
            punctionless = ''.join(punctionless)
            if self.stemming:
                lines.append([self.lem.stem(str(w)) for w in punctionless.split()])
            else:
                lines.append(punctionless.split())
            labels.append(df['stars'].loc[loc])
            business_id.append(df['business_id'].loc[loc])

        if self.binary == True:
            one_hot = [0.0, 0.0]
            hot_labels = []
            new_labels = []
            for label in labels:
                a=list(one_hot)
                if label <= 3:
                    new_labels.append(0)
                    a[0] = 1.0
                    hot_labels.append(a)
                else:
                    new_labels.append(1)
                    a[1] = 1.0
                    hot_labels.append(a)
            self.labels = new_labels
            self.y = np.array(hot_labels).reshape(-1, 1, self.n_classes)

        elif self.binary == False:
            one_hot = [0.0 for _ in range(self.n_classes)]
            hot_labels = []
            new_labels = []
            for label in labels:
                new_labels.append(int(label))
                a=list(one_hot)
                a[int(label)-1] = 1.0
                hot_labels.append(a)
            self.labels=new_labels
            self.y = np.array(hot_labels).reshape(-1, 1, self.n_classes)

        if self.padding:
            self.maxlen = max([len(line) for line in lines])
            for li in lines:
                if len(li) < self.maxlen:
                    for _ in range(self.maxlen-len(li)):
                        li.insert(0, nonce)


        self.reviews = lines
        self.business = business_id

    def word2id_builder(self):
        w2id_corpus=[]

        ct=0
        for line in self.reviews:
            for w in line:
                if self.stemming:
                    self.word2id[self.lem.stem(str(w))]=ct
                else:
                    self.word2id[str(w)]=ct
                ct+=1

            if self.stemming:
                w2id_corpus.append([self.word2id[self.lem.stem(str(w))] for w in line])
            else:
                w2id_corpus.append([self.word2id[str(w)] for w in line])
        self.word2id[nonce]=len(self.word2id)

        self.data = w2id_corpus


    def LSTM_trainer(self, model, with_embeds=False):

        for i in range(len(self.data)):
            if with_embeds:
                model.train_on_batch(x=np.array(self.data[i]).reshape(1, -1),
                                    y=np.array(self.labels[i]).reshape(1, -1, self.n_classes))
            else:
                model.train_on_batch(x=np.array(self.data[i]).reshape(1, -1, 1),
                                    y=np.array(self.labels[i]).reshape(1, -1, self.n_classes))

    def new_example(self, sentence):
        punctionless = [char for char in sentence if char not in string.punctuation]
        punctionless = ''.join(punctionless)
        strings = []
        for w in punctionless.split():
            if self.stemming:
                strings.append(self.lem.stem(str(w)))
            else:
                strings.append(str(w))

        hashed=[]
        for word in strings:
            try:
                hashed.append(self.word2id[word])
            except KeyError:
                hashed.append(self.word2id[nonce])

        if self.padding:
            if len(hashed) < self.maxlen:
                hashed.insert(0, self.word2id[nonce])
            elif len(hashed) > self.maxlen:
                hashed=hashed[:self.maxlen]

        return hashed
