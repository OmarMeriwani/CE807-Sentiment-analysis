import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from string import punctuation
from sklearn.preprocessing import scale
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()
from sklearn.feature_extraction.text import TfidfVectorizer

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def doc_to_clean_lines(doc, vocab):
    clean_lines = ''
    lines = doc.splitlines()

    for line in lines:
        tokens = line.split()
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
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
    mode = 'sentence' #all sentences or only full reviews (sentence,full)
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

print(train_docs)

training_tokens = [tokenizer.tokenize(t) for t in train_docs]
test_tokens = [tokenizer.tokenize(t) for t in test_docs]

w2v = Word2Vec(size=100, min_count=10)
w2v.build_vocab(sentences=training_tokens)
w2v.train(sentences=training_tokens,total_words=len(vocab),epochs=10)

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x for x in training_tokens])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_w2v = np.concatenate([buildWordVector(z, 100) for z in tqdm(map(lambda x: x, training_tokens))])
train_vecs_w2v = scale(train_vecs_w2v)
test_vecs_w2v = np.concatenate([buildWordVector(z, 100) for z in tqdm(map(lambda x: x, test_tokens))])
test_vecs_w2v = scale(test_vecs_w2v)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)
score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print (score[1])
