import pandas as pd
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os
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

def getNgram(tags,gram):
    counter = 0
    ngrams = []
    while counter < len(tags):
        if counter + gram <= len(tags) - 1:
            temp = []
            for i in range(counter, counter + gram):
                temp.append(tags[i])
            ngrams.append(''.join(temp))
            counter += 1
        else:
            break
    return ngrams
postrigrams = []
posquadgrams = []
pospetagram = []
for i in range(0,len(df)):
    sentence = df.loc[i][2]
    sentences = sent_tokenize(sentence)
    reviewPolarity = int(df.loc[i][3])
    tokens = []
    NER = scnlp.ner(sentence)
    POStagged = scnlp.pos_tag(sentence)
    POSTags = [p for word, p in POStagged]
    #print(POSTags)
    POSTriGrams = getNgram( POSTags,3)
    POSQuadriGrams = getNgram(POSTags,4)
    POSPentaGrams = getNgram(POSTags,5)
    POSTriGrams = [str(t).replace(',','').replace(':','') for t in POSTriGrams]
    POSQuadriGrams = [str(t).replace(',','').replace(':','') for t in POSQuadriGrams]
    POSPentaGrams = [str(t).replace(',','').replace(':','') for t in POSPentaGrams]
    if len(POSTriGrams) != 0:
        postrigrams.append([reviewPolarity,POSTriGrams])
    if len(POSQuadriGrams) != 0:
        posquadgrams.append([reviewPolarity,POSQuadriGrams])
    if len(POSPentaGrams) != 0:
        pospetagram.append([reviewPolarity,POSPentaGrams])
    if i > 1000:
        break
#Get POS from positive reviews
groups = 0
print(postrigrams)
print(posquadgrams)
print(pospetagram)
while groups < 3:
    PositivePOS = {}
    NegativePOS = {}

    posseq = []
    if groups == 0:
        posseq = postrigrams
    if groups == 1:
        posseq = posquadgrams
    if groups == 2:
        posseq = pospetagram
    for tt in posseq:
        for pt in tt[1]:
            if tt[0] > 2:
                if pt not in PositivePOS:
                    PositivePOS[pt] = 1
                else:
                    PositivePOS[pt] = PositivePOS.get(pt)  + 1
    #Get POS from negative reviews
            if tt[0] < 2:
                if pt not in NegativePOS:
                    NegativePOS[pt] = 1
                else:
                    NegativePOS[pt] = NegativePOS.get(pt)  + 1

    #Remove the intersection between the negative and positive POS
    intersection = [t for t,i in PositivePOS.items() if t in NegativePOS]
    for i in intersection:
        if i in PositivePOS:
            PositivePOS.pop(i)
    for i in intersection:
        if i in NegativePOS:
            NegativePOS.pop(i)
    filename = ''
    if groups == 0:
        filename = 'POSTrigrams.csv'
    if groups == 1:
        filename = 'POSQgrams.csv'
    if groups == 2:
        filename = 'POSPgrams.csv'

    with open(filename, 'w') as f:
        for key in PositivePOS.keys():
            f.write("%s,%s,%s\n"%(key,PositivePOS[key],1))
        for key in NegativePOS.keys():
            f.write("%s,%s,%s\n"%(key,NegativePOS[key],-1))

    groups += 1