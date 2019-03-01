import pandas as pd
import numpy as np
import nltk
import string
from nltk.stem import SnowballStemmer
import xml.etree.ElementTree as ET
import spacy

nlp = spacy.load('en')

class corpus_tools():

    def __init__(self):
        self.corpus = None
        self.labels = None
        self.word2id = {}
        self.lem = SnowballStemmer('english')
        super(corpus_tools, self).__init__()

    def line_corpus_from_dataframe(self, column, dataframe, stemming=False):
        lines=[]

        for loc in dataframe.index:
            if stemming:
                lines.append([self.lem.stem(str(w)) for w in nltk.sent_tokenize(dataframe[column].loc[loc])])
            else:
                lines.append(nltk.sent_tokenize(dataframe[column].loc[loc]))

        self.corpus = lines

    def yelp(self, binary=False):
        lines = []
        labels = []
        df = pd.read_csv('NLPhelperGEN/datasets/yelp.csv', skipinitialspace=True)
        for loc in df.index:
            text=df['text'].loc[loc]
            punctionless = [char for char in text if char not in string.punctuation]
            punctionless = ''.join(punctionless)
            lines.append(punctionless.split())
            labels.append(df['stars'].loc[loc])

        if binary:
            new_labels=[]
            for label in labels:
                if label <= 3:
                    new_labels.append(0)
                else:
                    new_labels.append(1)
            labels=new_labels

        self.corpus = lines
        self.labels = labels

    def columnized_corpus_from_dataframe(self, column, dataframe, stemming=False):
        lines = []

        for loc in dataframe.index:
            if stemming:
                for w in nltk.sent_tokenize(dataframe[column].loc[loc]):
                    lines.append([lem.stem(str(w)), loc])
            else:
                for w in nltk.sent_tokenize(dataframe[column].loc[loc]):
                    lines.append([str(w), loc])

        self.corpus = pd.DataFrame(np.array(lines).reshape(-1, 2), columns=['word', 'id'])
