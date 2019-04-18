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

def SentimentsPolarity(sentence):
    sn = SenticNet()
    values = []
    maxPolarity = 0.0
    prev = ''
    for stem in sentence:
        s = stemmer.stem(stem)
        try:
            polarity_value = sn.polarity_intense(s)
            pv = float(polarity_value)
            #print(prev)
            if prev in ['not','no','never']:
                pv = pv * -1
            '''sequels not fails makes 0 0 (0.136, 0.943)'''
            absPolarityVal = abs(pv)
            if absPolarityVal > abs(float(maxPolarity)):
                maxPolarity = pv
            values.append(pv)
        except Exception as e:
            prev = s
            continue
        prev = s
    if len(values) == 0:
        return 0.0
    avg = float(sum(values) / len(values)).__round__(3)
    return avg,maxPolarity



df_pbigrams = pd.read_csv('BigramsPolarity.csv',header=0,sep=',')
pbigrams = df_pbigrams.values.tolist()

df_punigrams = pd.read_csv('UnigramsPolarity.csv',header=0,sep=',')
punigrams = df_punigrams.values.tolist()

stop_words = stopwords.words('english')
stop_words = [s for s in stop_words if s not in ['no', 'not', 'never', 'n’t', 'nt']]

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
    sentenceClean = ' '.join([str(t).lower() for t in tokens if t not in stop_words])
    polarity2 = 0
    '''for pb in pbigrams:
        pbigram = pb[1] + ' ' + pb[2]
        if pbigram in sentenceClean:
            if pb[3] == 4:
                polarity2 += 1
            else:
                polarity2 -= 1
    polarity1 = 0
    for up in punigrams:
        punigram = str(up[1])
        if punigram in sentenceClean:
            if up[2] == 4:
                polarity1 += 1
            else:
                polarity1 -= 1
    '''
    normalStopwords = stopwords.words('english')

    cleanTokens = [str(t).lower() for t in tokens if t not in stop_words]
    #cleanTokensWithoutNegation = [str(t).lower() for t in tokens if t not in normalStopwords]
    #cleanTokensWithoutNegation = set(cleanTokensWithoutNegation)
    #Remove NER and choose specific POS Tags
    #usefultags = ['JJ','JJR','JJS','RB ','VB ','VBD','VBG','VBN','VBP','VBZ']
    #cleanTokensWithoutNegation = [t for t in cleanTokensWithoutNegation
    #                              if t not in
    #                              [str(n).lower() for n, nrt in NER if str(nrt).lower() != 'o'] and
    #                              t in [str(word).lower() for word, p in POStagged if p in usefultags ]]
    #ListOfCleanTokens.append([cleanTokensWithoutNegation, reviewPolarity])
    #avgAndmaxPol = SentimentsPolarity(cleanTokens)
    #print(sentenceClean, polarity2, polarity1,avgAndmaxPol)

'''
Get stopwords
lowercase
Separate sentences
POS
Senti
Negation fixing


Get stopwords
Get full sentence from each row with score
Remove stop words from each line
Convert to lower case
Remove punctuation
Get 1,2 words with scores
Build dictionary of 1,2 words with score
Sentence sentiments:
    * Get max polarity for each sentence considering negation after removing stop words
    * Create subj value according to the max polarity
Store 
-------------------
After preparing the list:
Tokenize
Remove stop words
Remove punctuation
Re-do term frequency
Divide terms/phrases into positive and negative
Remove the intersecting list between the two lists
Count the following:
    * The number of negative/positive in the document
    * The highest polarity of negative/positive words in the document
    * Sum of max polarity of document sentences by applying the last step on sentence level
-------------------
Load word2vec
Encode sentences

Build CNN
Get results

Build RNN
Get results

Use the result as a feature with the best model from above to predict with the statistical model

'''