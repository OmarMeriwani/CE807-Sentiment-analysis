import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
import os
from collections import Counter
from string import punctuation

def readfile(filename):
    df = pd.read_csv(filename,header=0,sep='\t')
    mode = 's'
    data = ''
    prev = ''
    for i in range(0,len(df)):
        if mode != 'all':
            if prev != str(df.loc[i][1]):
                sentence = df.loc[i][2]
                prev = str(df.loc[i][1])
            else:
                continue
        else:
            sentence = df.loc[i][2]
        data = data + '.' + sentence
    return data
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

data = readfile('train.csv')
data2 = readfile('test.csv')
data = data +' '+  data2
tokens = clean_doc(data)
vocabulary = Counter()
vocabulary.update(tokens)
items = [word for word, count in vocabulary.items() if count >= 2]
items = '\n'.join(items)
file = open('vocabulary.txt', 'w')
file.write(items)
file.close()

