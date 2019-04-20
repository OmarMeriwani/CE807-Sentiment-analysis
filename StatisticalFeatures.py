import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import seaborn
from sklearn.model_selection import train_test_split
from ast import literal_eval
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

df  =pd.read_csv('TrainingDataset.csv', header=0, sep=',')
train, test = train_test_split(df, test_size=0.2)
columns = ['BigramsPolarity','UnigramsPolarity','SenticnetAVG','senticnetMAX','WordsInScore','POSSequenceScore']
x_train = list(np.array(train)[:, [3,4,6]])
y_train = train['y']
#5 is bad, 7 is bad
x_test = list(np.array(test)[:,[3,4,6]])
y_test = test['y']

nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)
print(len(y_pred))
print(len(y_test))

print("\n====================Kappa Statistic====================\n", cohen_kappa_score(y_test, y_pred) )
print("\n====================Confusion Matrix====================\n", pd.crosstab(y_test, y_pred))
print("\n====================Precision table====================\n", classification_report(y_test, y_pred))
print("\n====================Accuracy====================\n ", accuracy_score(y_test, y_pred))