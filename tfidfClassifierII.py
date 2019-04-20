import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from string import punctuation
from sklearn.preprocessing import scale
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from stanfordcorenlp import StanfordCoreNLP
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os
from nltk.stem.porter import *

java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path
host='http://localhost'
port=9000
scnlp =StanfordCoreNLP(host, port=port,lang='en', timeout=30000)
stemmer = PorterStemmer()


tqdm.pandas(desc="progress-bar")
#stemmer = SnowballStemmer("english")
stop_words = stopwords.words('english')

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def doc_to_clean_lines(doc, vocab):
    clean_lines = ''
    lines = doc.splitlines()

    for line in lines:
        # split into tokens by white space
        tokens = line.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w.lower() in vocab]
        clean_lines = ' '.join(tokens)
    return clean_lines

# load the vocabulary
#
vocab_filename = 'vocabulary.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
vocab = [v.lower() for v in vocab]


def readfile(filename):
    df = pd.read_csv(filename,header=0,sep='\t')
    mode = 'all' #all sentences or only full reviews (sentence,full)
    data = []
    prev = ''
    for i in range(0,len(df)):
        if mode == 'sentence':
            if prev != str(df.loc[i][1]):
                sentence = df.loc[i][2]
                prev = str(df.loc[i][1])
            else:
                continue
        else:
            sentence = df.loc[i][2]
        reviewPolarity = int(df.loc[i][3])
        '''tokens = []
        tknzr = RegexpTokenizer(r'\w+')
        t = tknzr.tokenize(sentence)
        for tk in t:
            tokens.append(tk)
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]'''
        sentence = doc_to_clean_lines(sentence,vocab)
        data.append([sentence,reviewPolarity])
    return data

def split(docs, percentage):
    length = len(docs)
    firstlength = int (length * percentage)
    training = docs[:firstlength]
    test = docs[firstlength:length]
    return training,test
def tfidf(docs):
    tfidf = TfidfVectorizer(analyzer='word',
                            strip_accents = 'ascii',
                            encoding='utf-8',
                            ngram_range = (1,2,3),
                            min_df = 3,
                            sublinear_tf = True)
    train_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in docs]
    train_df_flags = tfidf.transform(train_phrases2)
    return train_df_flags
lemmatizer = WordNetLemmatizer()

def preprocess(docs):
    result = []
    for i in docs:
        tokens = tokenizer.tokenize(i)
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [stemmer.stem(str(t).lower()) for t in tokens]
        # tokens = [t for t in tokens if t not in stop_words]
        result.append(' '.join(tokens))
    result = list(result)
    return result
def postags(docs):
    result = []
    for i in docs:
        postags_list = []
        for word,postag in scnlp.pos_tag(i):
            postags_list.append(postag)
        result.append(' '.join(postags_list))
    result = list(result)
    return result

data = np.array(readfile('train.csv'))
print(data.shape)
#print(data[:,0])
traindata, testdata = split(data,0.7)
print(testdata.shape)
print(traindata.shape)
train_docs = traindata[:,0]
test_docs = testdata[:,0]
y_train = traindata[:,1]
y_test = testdata[:,1]
x_train = preprocess(train_docs)
x_test = preprocess(test_docs)

tfidf = TfidfVectorizer(analyzer='word')
#print(x_train)

_ = tfidf.fit(x_train)
train_tfidf = tfidf.transform(x_train)
test_tfidf = tfidf.transform(x_test)

print('TYPE: ',type(train_tfidf))
model1 = LinearSVC()
model1.fit(train_tfidf,y_train)
print(model1.score(test_tfidf,y_test))


df = pd.read_csv('test.csv', header=0, sep='\t')
new_df = pd.DataFrame(columns=['PhraseId','SentenceId',	'Phrase', 'Sentiment'])
counter = 0
tfidiftest = []
tftest = []
for i in range(0, len(df)):
    sentence = df.loc[i][2]
    tokens = tokenizer.tokenize(sentence)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(str(t).lower()) for t in tokens]
    sentence = ' '.join(tokens)
    tftest.append(sentence)
tf = tfidf.transform(tftest)
for i in tf:
    tfidiftest.append(tf)
for i in range(0, len(df)):
    pred = model1.predict(tfidiftest[i])
    new_df.loc[counter] = [df.loc[i][0],df.loc[i][1],df.loc[i][2],pred]
    counter += 1
new_df.to_csv('test_new.csv',sep='\t')
'''
Conv:
Word2vec Google news: 25%
Word2vec IMDB: 24%
Word2vec same text: 25%
Statistical Symantical Features:
37%
Linear SVC:
TFIDF only sentences: 37%
TFIDF all sentences with stemmer: 57%
TFIDF all sentences without stems: 52%
TFIDF all sentences with stems and lemma: 57%
TFIDF all sentences with porter stemmer and lemma: 57%
TFIDF all sentences with porter stemmer: 57%
TFIDF all sentences with porter stemmer keeping stopwords: 58%
MLPClassifier:
NB:
TFIDF all sentences with porter stemmer keeping stopwords: 54%
LogisticRegression:
TFIDF all sentences with porter stemmer keeping stopwords: 58%
POS Tags 50%
'''