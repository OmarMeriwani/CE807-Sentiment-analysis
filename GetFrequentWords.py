import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
import os
from senticnet.senticnet import SenticNet
from nltk.stem.porter import *

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)
stemmer = PorterStemmer()

stop_words = stopwords.words('english')

df = pd.read_csv('train.csv',header=0,sep='\t')
prev = ''
tknzr = RegexpTokenizer(r'\w+')
ListOfCleanTokens = []
for i in range(0,len(df)):
    if prev != str(df.loc[i][1]):
        sentence = df.loc[i][2]
        prev = str(df.loc[i][1])
    else:
        continue
    sentences = sent_tokenize(sentence)
    reviewPolarity = int(df.loc[i][3])
    tokens = []
    for sent in sentences:
        t = tknzr.tokenize(sent)
        for tk in t:
            tokens.append(tk)
    NER = scnlp.ner(sentence)
    POStagged = scnlp.pos_tag(sentence)
    normalStopwords = stopwords.words('english')

    cleanTokensWithoutNegation = [str(t).lower() for t in tokens if t not in normalStopwords]
    cleanTokensWithoutNegation = set(cleanTokensWithoutNegation)
    #Remove NER and choose specific POS Tags
    usefultags = ['JJ','JJR','JJS','RB ','VB ','VBD','VBG','VBN','VBP','VBZ']
    cleanTokensWithoutNegation = [t for t in cleanTokensWithoutNegation
                                  if t not in
                                  [str(n).lower() for n, nrt in NER if str(nrt).lower() != 'o'] and
                                  t in [str(word).lower() for word, p in POStagged if p in usefultags ]]
    ListOfCleanTokens.append([cleanTokensWithoutNegation, reviewPolarity])

GoodWords = {}
BadWords = {}
for i in ListOfCleanTokens:
    cleanTokens = i[0]
    reviewPolarity = i[1]
    if reviewPolarity > 2:
        for t in cleanTokens:
            if t not in GoodWords:
                GoodWords[t] = 1
            else:
                GoodWords[t] = GoodWords.get(t) + 1
    if reviewPolarity < 2:
        for t in cleanTokens:
            if t not in BadWords:
                BadWords[t] = 1
            else:
                BadWords[t] = BadWords.get(t) + 1

intersection = [t for t,i in GoodWords.items() if t in BadWords]
for i in intersection:
    if i in GoodWords:
        GoodWords.pop(i)
for i in intersection:
    if i in BadWords:
        BadWords.pop(i)

print(GoodWords)
print(BadWords)
import csv
with open('PositiveWords.csv', 'w') as f:
    for key in GoodWords.keys():
        f.write("%s,%s\n"%(key,GoodWords[key]))
with open('NegativeWords.csv', 'w') as f:
    for key in BadWords.keys():
        f.write("%s,%s\n"%(key,BadWords[key]))
