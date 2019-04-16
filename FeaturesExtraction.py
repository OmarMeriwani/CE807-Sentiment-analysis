import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import nltk
import os
#nltk.download('stopwords')
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path

stop_words = stopwords.words('english')
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)

df = pd.read_csv('train.csv',header=0,sep='\t')
print(stop_words)
prev = ''
for i in range(0,len(df)):
    sentence = ''
    if prev == str(df.loc[i][1]):
        sentence = df.loc[i][2]
    else:
        prev = str(df.loc[i][1])
        continue
    POStagged = scnlp.pos_tag(sentence)
    for p in POStagged:
        print(p[0])
'''
for i in range(0,len(df)):
    if prev != str(df.loc[i][1]):
        sentence = df.loc[i][2]
        prev = str(df.loc[i][1])
    else:
        continue
    sentences = sent_tokenize(sentence)
    tokens = []
    for sent in sentences:
        t = scnlp.word_tokenize(sent)
        for tk in t:
            tokens.append(tk)
    NER = scnlp.ner(sentence)
    POStagged = scnlp.pos_tag(sentence)'''


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