import pandas as pd
import numpy as np

df = pd.read_csv('train.csv',header=0,sep='\t')

for i in range(0,len(df)):
    print(df.loc[i][2])

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