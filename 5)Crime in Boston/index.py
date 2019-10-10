import numpy as np 
import pandas as pd
import seaborn as sns 
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("crime.csv")
del df['SHOOTING']
df.dropna(inplace=True)
df.isnull().sum()

r = {}
for x,y in zip(df["OFFENSE_CODE_GROUP"].value_counts().index,df["OFFENSE_CODE_GROUP"].value_counts()):
    r[x] = y
df["Count_offense"] = [r[x] for x in df["OFFENSE_CODE_GROUP"]]
cat = [df["OFFENSE_CODE_GROUP"].value_counts().index[x] for x in range(9)]
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
for var, subplot in zip(cat, ax.flatten()):
    df[df["OFFENSE_CODE_GROUP"]==var]["DISTRICT"].value_counts().plot(kind="bar",ax=subplot)
    subplot.set_ylabel(var)
fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)


cat1 = [df["DISTRICT"].value_counts().index[x] for x in range(12)]
r = pd.DataFrame()
r["OFFENSE_CODE_GROUP"],r["DISTRICT"],cat =  df["OFFENSE_CODE_GROUP"],df["DISTRICT"],[df["OFFENSE_CODE_GROUP"].value_counts().index[x] for x in range(9,63)]
r["MONTH"],r["YEAR"] = df["MONTH"],df["YEAR"] 
r["OCCURRED_ON_DATE"]= df["OCCURRED_ON_DATE"]
for x in cat:
    r.drop(r[r["OFFENSE_CODE_GROUP"]==x].index,inplace=True)
cat11,cat12 = cat1[:len(cat1)//2],cat1[len(cat1)//2:]

#Convert str to timestamp
# df["OCCURRED_ON_DATE"] = pd.to_datetime(pd.Series(df["OCCURRED_ON_DATE"]))
type(r["OCCURRED_ON_DATE"][0])

r["OCCURRED_ON_DATE"] = df["OCCURRED_ON_DATE"]
r["OCCURRED_ON_DATE"]=[x[:10] for x in r["OCCURRED_ON_DATE"]]
r["OCCURRED_ON_DATE"]=pd.to_datetime(pd.Series(r["OCCURRED_ON_DATE"]))
r["count"] = [1 for x in r["OCCURRED_ON_DATE"]]
res = r.pivot_table(index="OCCURRED_ON_DATE",columns="OFFENSE_CODE_GROUP",values="count",aggfunc='sum')

res.dropna(inplace=True)
res = pd.DataFrame(res.to_records())
res.plot()

f = res.set_index('OCCURRED_ON_DATE')
cat3 = f.columns
cat31,cat32,cat33 = cat3[:3],cat3[3:6],cat3[6:]
y = f['Larceny'].resample('MS').mean()


for x in cat31:
     y = f[x].resample('MS').mean()
     y.plot(figsize=(15, 6))
     plt.legend(loc = x)
plt.show()
    


"""
Obviously, the 9 most common type of crimes in boston are :
Motor Vehicle Accident Response
Larceny
Medical assistance
Investigate person
Other
Simple Assault
Vandalism
Drug Violation
Verbal Disputes
these crimes account for more than 50% of boston crimes
 
2)64.3% of the crime in boston occur in the Districts [B2,C11,D4,B3,A1], 
so we can conclude that these district are where the most bostom crimes took place, 
Also, the barcharts tells us that most of the crimes in Boston took place
in district B2 and C11 except for the thefts that appear most in district D4 and A1

3)
We notice that the Larency rate in Boston reaches its maximum between June and August and decrease gradually over the years.
that is the summer, while the rate of drug-violation decreases remarkably at the end of each years.
About the investigate person crime, the rate peaks in summer and increase progressively over the years.

Motor-vehicle-Accident increase in the end of the spring and decrease at the end of each years.
Medical-Assistance increase progressively over the years and for some reason,
this crime increase exponentially and reaches its peak in June 2018.

Verbal-Disputes rate increase gradually over the year and reach its peak in the middle of the years.
Simple-assault rate has not changed much but for some reason increase exponentially 
in march 2018 until it reaches its maximum in may and decrease as fast as it peaks. 
Vandalism rate peak and valley gradually until he stats decreasing in august 2017.

The last plot of time-series analysis represent the distribution of all the crime's frequency
in boston,  we can not deny the fact that the majority of the crimes decrease progressively 
over the year except for (Investigate-person, Medical Assistance, Verbal Disputes).

So to answer our question, according to our observation the frequency of crimes change over the days.
 

"""


res1 = r.pivot_table(index="OCCURRED_ON_DATE",columns="DISTRICT",values="count",aggfunc='sum')
res1.dropna(inplace=True)
res1 = pd.DataFrame(res1.to_records())
f = res1.set_index('OCCURRED_ON_DATE')
f.index
cat3 = f.columns
cat31,cat32,cat33,cat34 = cat3[:3],cat3[3:6],cat3[6:9],cat3[9:12]

for x in cat31:
     y = f[x].resample('MS').mean()
     y.plot(figsize=(15, 6))
     plt.legend(loc = x)
plt.show()












 