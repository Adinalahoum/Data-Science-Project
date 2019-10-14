import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df['Age'].fillna(df['Age'].mode())

del df["Cabin"],df["PassengerId"]

df["Age"].fillna(df["Age"].mode()[0], inplace=True)
df.dropna(inplace=True)
df.isnull().sum()
#df.pivot_table(values='price',index=[x],columns=['price_categorie'],aggfunc =lambda x: len(x.unique()))

df["Pclass"].value_counts().plot(kind = "pie", autopct='%1.1f%%')

df["Survived"] = np.where(df["Survived"]==1, 'yes', 'no')

sns.countplot(x = "Pclass",hue="Survived",data = df)

df["Related"] = df["SibSp"]+df["Parch"]

df.pivot_table(values='Related',index=["Survived"],columns=["Pclass"],aggfunc =lambda x: len(x.unique())).plot(kind="bar") 
plt.ylabel("Related")




from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df["Embarked"] = df["Embarked"].replace({"S":0,"C":1,"Q":2},regex=True)
df["Sex"] = df["Sex"].replace({"female":0,"male":1})
feature_cols = ["Embarked","Sex"]
X = df[feature_cols]
Y = df["Survived"]
logistic_model = LogisticRegression(random_state = 0,solver='liblinear')
logistic_model.fit(X,Y)

df_test["Embarked"] = df_test["Embarked"].replace({"S":0,"C":1,"Q":2},regex=True)
df_test["Sex"] = df_test["Sex"].replace({"female":0,"male":1})
df_test["Survived_lgr_proba"] = logistic_model.predict_proba([df_test[feature_cols]])
Survived_pred_LgR = logistic_model.predict(df_test[feature_cols])

df_test["Survived_lgr"] = Survived_pred_LgR
df_test["Survived_lgr_proba"] = logistic_model.predict_proba(df_test[feature_cols])

Rdf=RandomForestClassifier(n_estimators=10)
Rdf.fit(X,Y)

Survived_pred_Rdf = Rdf.predict(df_test[feature_cols])
df_test["Survived_Rdf"] = Survived_pred_Rdf

r = df_test.loc[df_test["Survived_Rdf"]!=df_test["Survived_lgr"]]
len(r)

df_test["Embarked"] = df_test["Embarked"].replace({0:"S",1:"C",2:"Q"})
df_test["Sex"] = df_test["Sex"].replace({0:"female",1:"male"})

df_test.isnull().sum()


res = df_test.index[df_test["Survived_Rdf"]==df_test["Survived_lgr"]].tolist()
len(res)





#Conclusion 























 


















