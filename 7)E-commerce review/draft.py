import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np 
import pandas as pd
import seaborn as sns

df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

del df["Unnamed: 0"]

res = [str(x) for x in df["Review Text"]]

df["Review Text"] = res

print(res[2])

sns.catplot(x="Clothing ID", kind="count", data=df,height=8, aspect=1)

train = df.dropna(inplace=True)



from collections import Counter
Counter(" ".join([str(x) for x in df["Review Text"].loc[df["Rating"]<=2]]).split()).most_common(100)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

l = []
for x in res:
    for y in word_tokenize(x):
        l.append(y)
l = list(dict.fromkeys(l))
sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neu_word_list=[]
neg_word_list=[]

pos = 0
neg = 0
for word in res[2].split(" "):
    if (sid.polarity_scores(word)['compound']) >= 0.5:
        pos+= 1 
    elif (sid.polarity_scores(word)['compound']) <= -0.5:
        neg+=1
print('Positive: ' + str(float(pos)/len(res[2])))
print('Negative: ' + str(float(neg)/len(res[2])))

print('Positive :',pos_word_list)        
print('Neutral :',neu_word_list)    
print('Negative :',neg_word_list) 

r = "Dress runs small esp where the zipper area runs. i ordered the sp which typically fits me and it was very tight! the material on the top looks and feels very cheap that even just pulling on it will cause it to rip the fabric. pretty disappointed as it was going to be my christmas dress this year! needless to say it will be going back."














