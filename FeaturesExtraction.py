import pandas as pd
import numpy as np

df = pd.read_csv('train.csv',header=0,sep='\t')

for i in range(0,len(df)):
    print(df.loc[i][2])