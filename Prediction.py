import pandas as pd

df = pd.read_csv('submission.csv',sep='\t')
df_new = pd.DataFrame(columns=['PhraseId', 'Sentiment'])
for i in range(0, len(df)):
    df_new.loc[i] = [df.loc[i][1],df.loc[i][4]]
df_new.to_csv('submission2.csv',index=False)