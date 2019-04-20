import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
import nltk.tokenize

stemmer = SnowballStemmer("english")

def read_file():
    df_train1 = pd.read_csv("train.csv", sep='\t')
    df_test = pd.read_csv("test.csv", sep='\t')
    # print(df_train1.head)
    return df_train1, df_test

def cleaning(self,s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = re.sub(r'&amp;', '', s)
    s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = re.sub(r'[^\x00-\x7f]',r'',s)
    s = re.sub(r'\'', ' ', s)
    return s

def stem_lemmatize(self,df_train,df_test):
    sentences = list(df_train.Phrase.values) + list(df_test.Phrase.values)
    sentences2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in sentences]
    #sentences2 = [[lem.lemmatize(word) for word in sentence.split(" ")] for sentence in sentences]
    for i in range(len(sentences2)):sentences2[i] = ' '.join(sentences2[i])
    #print(len(sentences2))
    return sentences2
def tfidf(self,df_train,df_test,sentences2):
    tfidf = TfidfVectorizer(analyzer='word',
                            strip_accents = 'ascii',
                            encoding='utf-8',
                            ngram_range = (1,2),
                            min_df = 3,
                            sublinear_tf = True)
    _ = tfidf.fit(sentences2)
    print(len(tfidf.get_feature_names()))
    train_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_train.Phrase.values)]
    for i in range(len(train_phrases2)):train_phrases2[i] = ' '.join(train_phrases2[i])
    train_df_flags = tfidf.transform(train_phrases2)
    test_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_test.Phrase.values)]
    for i in range(len(test_phrases2)):test_phrases2[i] = ' '.join(test_phrases2[i])
    test_df_flags = tfidf.transform(test_phrases2)
    return train_df_flags,test_df_flags

def split(docs, percentage):
    length = len(docs)
    firstlength = int (length * percentage)
    training = docs[:firstlength]
    test = docs[firstlength:length]

    return training,test

def train_test_split(train_df_flags,df_train):
    X_train_tf = train_df_flags[0:6000]
    X_valid_tf = train_df_flags[6000:]
    y_train_tf = (df_train["Sentiment"])[0:6000]
    y_valid_tf = (df_train["Sentiment"])[6000:]
    return  X_train_tf,X_valid_tf,y_train_tf,y_valid_tf

training, testing = read_file()
print(training.values[:,2])
trainingtokens = [stemmer.stem(nltk.tokenize.word_tokenize(t)) for t in training[:,2]]
tesingtokens = [stemmer.stem(nltk.tokenize.word_tokenize(t)) for t in testing[:,2]]
testing[:,2] = [cleaning(s) for s in testing[:,2]]
sentences = stem_lemmatize(training,testing)
train_df_flags, test_df_flags = tfidf(training, testing, sentences)  # tfidf
X_train,X_test,y_train,y_test = train_test_split(train_df_flags,training)

svc = LinearSVC(dual=False)
svc.fit(X_train, y_train)
print("Accuracy of LinearSVC:", svc.score(X_test, y_test))
scores = cross_val_score(svc, X_train, y_train, cv=5)
print("Cross validation scores: ", scores)
