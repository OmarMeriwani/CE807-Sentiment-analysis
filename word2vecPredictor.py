import numpy as np
from string import punctuation
from os import listdir
from numpy import array
import pandas as pd
from numpy import zeros
from numpy import asarray
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def process_docs2(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines

# load the vocabulary
vocab_filename = 'D:/aclImdb/imdb.vocab'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
def load_embedding(filename):
	# load embedding into memory, skip first line
	file = open(filename,'r')
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding
def get_weight_matrix(embedding, vocab):
	# total vocabulary size plus 0 for unknown words
	vocab_size = len(vocab) + 1
	# define weight matrix dimensions with all 0
	weight_matrix = zeros((vocab_size, 100))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = embedding.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix
def readfile(filename):
	df = pd.read_csv(filename,header=0,sep='\t')
	mode = 's'
	data = []
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
		reviewPolarity = int(df.loc[i][3])
		'''tokens = []
		tknzr = RegexpTokenizer(r'\w+')
		t = tknzr.tokenize(sentence)
		for tk in t:
			tokens.append(tk)
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]'''
		data.append([sentence,reviewPolarity])
	return data
data = np.array(readfile('train.csv'))
print(data.shape)
#print(data[:,0])
traindata = data[:6000]
testdata = data[6000:8600]
print(testdata.shape)
print(traindata.shape)
train_docs = traindata[:,0]
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)

'''==============================================='''
# pad sequences
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define training labels
ytrain = traindata[:,1]
'''==============================================='''
# create the tokenizer
data = np.array(readfile('train.csv'))

# load all test reviews
test_docs = testdata[:,0]
'''==============================================='''
# pad sequences
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)
encoded_docs = tokenizer.texts_to_sequences(test_docs)

Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# define test labels
ytest = testdata[:,1]

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1

# load embedding from file
raw_embedding = load_embedding('imdb_word2vec.txt')
# get vectors in the right order
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
# create the embedding layer
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=2)
print('Test Accuracy: %f' % (acc*100))