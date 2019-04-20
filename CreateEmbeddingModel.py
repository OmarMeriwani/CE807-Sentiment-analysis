from string import punctuation
from os import listdir
from gensim.models import Word2Vec
import pandas as pd

def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text


def doc_to_clean_lines(doc, vocab):
	clean_lines = list()
	lines = doc.splitlines()
	for line in lines:
		# split into tokens by white space
		tokens = line.split()
		# remove punctuation from each token
		table = str.maketrans('', '', punctuation)
		tokens = [w.translate(table) for w in tokens]
		# filter out tokens not in vocab
		tokens = [w for w in tokens if w in vocab]
		clean_lines.append(tokens)
	return clean_lines
def readfile(filename):
	df = pd.read_csv(filename,header=0,sep='\t')
	mode = 'sentence' #all sentences or only full reviews (sentence,full)
	data = ''
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
		data += '\n' + (sentence)
	return data

def process_docs2(directory, vocab):
	lines = list()
	doc = readfile(directory)
	doc_lines = doc_to_clean_lines(doc, vocab)
	lines += doc_lines
	return lines
vocab_filename = 'vocabulary.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
docs = process_docs2('train.csv',vocab)
#docs2 = process_docs2('test.csv',vocab)
sentences =  docs
model = Word2Vec(sentences, window=5, workers=8, min_count=1)
words = list(model.wv.vocab)
filename = 'embedding2.txt'
model.wv.save_word2vec_format(filename, binary=False)
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(train_docs)
#encoded_docs = tokenizer.texts_to_sequences(train_docs)
#max_length = max([len(s.split()) for s in train_docs])
#Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')