import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
import os
#nltk.download('stopwords')
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
import string
stop_words = stopwords.words('english')
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('train.csv',header=0,sep='\t')
print(stop_words)
prev = ''
unigrams = pd.DataFrame(columns=['word','polarity'])
bigrams = pd.DataFrame(columns=['word1','word2','polarity'])
counter1 = 0
counter2 = 0
stop_words = [s for s in stop_words if s not in ['no', 'not', 'never', 'n’t', 'nt']]

for i in range(0,len(df)):
    sentence = ''
    polarity = df.loc[i][3]
    if prev == str(df.loc[i][1]):
        sentence = df.loc[i][2]
    else:
        prev = str(df.loc[i][1])
        continue
    POStagged = scnlp.pos_tag(sentence)
    tknzr = RegexpTokenizer(r'\w+')
    tokens = tknzr.tokenize(sentence)
    tokensWithoutStopWords = [str(t).lower() for t in tokens if t not in stop_words]
    LengthOfSentence = len(tokens)
    if polarity ==0 or polarity == 4:
        if len(tokensWithoutStopWords) == 1:
            unigrams.loc[counter1] = [tokensWithoutStopWords[0],polarity]
            counter1 += 1
        if len(tokensWithoutStopWords) == 2:
            bigrams.loc[counter2] = [tokensWithoutStopWords[0],tokensWithoutStopWords[1],polarity]
            counter2 += 1
        print(i)
unigrams.to_csv('UnigramsPolarity.csv')
bigrams.to_csv('BigramsPolarity.csv')
