from string import punctuation
from os import listdir
from gensim.models import Word2Vec

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
def process_docs2(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		doc = load_doc(path)
		doc_lines = doc_to_clean_lines(doc, vocab)
		# add lines to list
		lines += doc_lines
	return lines
vocab_filename = 'D:/aclImdb/imdb.vocab'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
positive_docs = process_docs2('D:/aclImdb/train/pos', vocab, True)
negative_docs = process_docs2('D:/aclImdb/train/neg', vocab, True)
positive_docs2 = process_docs2('D:/aclImdb/test/pos', vocab, True)
negative_docs2 = process_docs2('D:/aclImdb/test/neg', vocab, True)
sentences = negative_docs + positive_docs + positive_docs2 + negative_docs2
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
words = list(model.wv.vocab)
filename = 'imdb_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(train_docs)
#encoded_docs = tokenizer.texts_to_sequences(train_docs)
#max_length = max([len(s.split()) for s in train_docs])
#Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')