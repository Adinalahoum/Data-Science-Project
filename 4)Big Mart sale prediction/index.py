#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn as sk
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

#Load the dataset
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
train['source']='train'
test['source']='test'
df = pd.concat([train, test],ignore_index=True)

#1)Some data cleaning
#Visualize some features of the dataset
categorical_columns = [df[x].value_counts() for x in df.dtypes.index if df.dtypes[x]=='object']
for x in categorical_columns:
	print(x)

#Cleaning some data
df = df.replace(to_replace =["LF","low fat"], value ="Low Fat" )
df = df.replace(to_replace ="reg" , value ="Regular")

#Count the Na value
df.isnull().sum()


#Define function that allow us to fill na value based on another column
def fill_na_based_on_another_column(df,x,y):
	dff = df[[x,y]].dropna()
	dic,k,j,i= dict(zip(dff[x],dff[y])),df.columns.get_loc(x),df.columns.get_loc(y),0
	for a in df[y]:
		if math.isnan(float(a)):
			if df.iloc[i,k] in dic.keys():
				df.iloc[i,j] = dic[df.iloc[i,k]]
			else :
				df.iloc[i,j] = df[y].mean()
		i+=1

#filling the na value
fill_na_based_on_another_column(df,"Item_Identifier","Item_Weight")
df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())
t = df['Outlet_Size'].dropna().unique().tolist()
df["Outlet_Size"] = df["Outlet_Size"].replace(t,[0,1,2])
df["Outlet_Size"] = df["Outlet_Size"].apply(lambda x: np.random.choice(df["Outlet_Size"].dropna().values) if np.isnan(x) else x)
df["Outlet_Size"] = df["Outlet_Size"].replace([0,1,2],t)

#no more mising value
df.isnull().sum()

#Modify Item_visibility
sum(df['Item_Visibility'] == 0)
df['Item_Visibility'] = df['Item_Visibility'].replace(0,np.nan)
df['Item_Visibility']=df['Item_Visibility'].fillna(df['Item_Visibility'].mean())

#Edit the variable Item_Type
df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
df['Item_Type_Combined'].value_counts()
df.loc[df['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
df['Item_Fat_Content'].value_counts()

#Create a new variable depicting the years of operation of a store.
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
df['Outlet_Years'].describe()



#2)Visualization
#numeric data
num = df._get_numeric_data().drop('Outlet_Establishment_Year',1)[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Years','Item_Outlet_Sales']]
num.describe()

#Correlation matrix
corr = num.corr()
corr.style.background_gradient(cmap='coolwarm')

#categorical data
cat = df.select_dtypes(include='object').drop(["Item_Identifier","Outlet_Identifier"],1)
fig, ax = plt.subplots(2,3, figsize=(20, 10))
for variable, subplot in zip(cat, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)

cat["Item_Outlet_Sales"] = df["Item_Outlet_Sales"]
cat['Outlet_Identifier'] = df['Outlet_Identifier']

sns.catplot(y="Item_Type",x="Item_Outlet_Sales", data=cat,kind = "bar",palette="pastel");
sns.catplot(y="Outlet_Identifier",x="Item_Outlet_Sales", hue="Item_Type_Combined", kind="bar", data=cat);
sns.catplot(y="Outlet_Identifier",x="Item_Outlet_Sales", hue="Outlet_Location_Type", kind="bar", data=cat);
sns.catplot(y="Outlet_Identifier",x="Item_Outlet_Sales", hue="Outlet_Size", kind="bar", data=cat);

#3)Building the model
#One hot encoding
le = LabelEncoder()
#New variable for outlet
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

#Drop the columns which have been converted to different types:
df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = df.loc[df['source']=="train"]
test = df.loc[df['source']=="test"]


#Define a function which create the model, visualization and cross validation
#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
def modelfit(alg, dtrain, dtest, predictors, target, IDcol):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score =  cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20)
    cv_score = np.sqrt(np.abs(cv_score))

    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))



#Linear Regression Model
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol)
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

#Ridge Regression Model
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')




