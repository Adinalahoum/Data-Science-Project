import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np

#Load the data
df = pd.read_csv("dataset.csv")
#1)Preparation et nettoyage des données
#eliminer la variable car_id parceque c'est pas nécessaire
df = df.drop(columns="car_ID")
#Avoir une idée sur la dataset
df.head()
#Separer et nettoyer la colonne CarName
Carcompany,Carmodel,a,b = [],[],"",""
for x in df["CarName"]:
	if len(x.split(" ",1))==1:
		a = x
		b = ""
	else:
		a,b = x.split(" ",1)
	Carcompany.append(a)
	Carmodel.append(b)
df["Carcompany"]=Carcompany
df["Carmodel"]=Carmodel
df["Carcompany"].value_counts()
[df.count()-205]
df = df.replace(to_replace =["vokswagen","vw"], value ="volkswagen" )
df = df.replace(to_replace ="Nissan" , value ="nissan" )
df = df.replace(to_replace = "toyouta", value ="toyota" )
df = df.replace(to_replace = "porcshce", value = "porsche")
df = df.reindex(columns=["symboling", "CarName","Carcompany","Carmodel", "fueltype", "aspiration", "doornumber", "carbody","drivewheel", "enginelocation", "wheelbase", "carlength", "carwidth","carheight", "curbweight", "enginetype", "cylindernumber", "enginesize", "fuelsystem", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg","price","price_categorie"])
df = df.replace(to_replace ="4wd" , value ="rwd" )
#Creer une nouvelle variable de prix pour mieux visualiser les données Aprés
r = []
for x in df["price"]:
	if x >18000:
		r.append("cher")
	elif x<18000 and x>10000:
		r.append("moyen")
	elif x<10000:
		r.append("abordable")
df["price_categorie"] = r
#Avoir une idée sur la statistique descriptive des variables numériques
numerical_all = ["wheelbase","carlength","carwidth","carheight","curbweight","enginesize","boreratio","stroke", "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg","price"]
df[numerical_all].describe()

#Visualisation de la matrice de correlation
cor = df[numerical_all].corr()
sns.heatmap(cor,xticklabels=cor.columns,yticklabels=cor.columns,annot = True)

#Visualization of some variables correlated with the price
numerical = ["carlength","carwidth","enginesize","horsepower","curbweight","price"]
df[numerical].hist(bins=15, figsize=(15, 6), layout=(3, 3))
sns.pairplot(df[numerical])
sns.distplot(df["price"])

#Get the summary of categorical variable

#Countplot
categorical = ["Carcompany","aspiration","drivewheel","carbody","fueltype","doornumber","enginetype","cylindernumber","fuelsystem"]
fig, ax = plt.subplots(3,3, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)


#Create a pivot_table
def PV(x):
	d = df.pivot_table(values='price',index=[x],columns=['price_categorie'],aggfunc =lambda x: len(x.unique()))
	return d.reindex(columns=["abordable","moyen","cher"])

#Visualize the pivot_table
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    PV(variable).plot(kind="bar", ax=subplot,legend=None)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
fig.subplots_adjust(left=0.2, wspace=0.4, hspace = 0.6)

asp = PV('cylindernumber')
#fig.savefig('pivot_table')
symboling,aspiration,drivewheel,carbody,fueltype,doornumber,enginetype,cylindernumber,fuelsystem =[PV(x) for x in categorical]

#2)Create the model:
#fonction pour construire un modele de regression
def regression(X,Y,title,xlabel,ylabel="prix"):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
	r = LinearRegression()
	r.fit(X_train, y_train)
	y_pred = r.predict(X_test)
	print('R2 score: %.2f' % r2_score(y_test,y_pred))
	plt.scatter(X_test, y_test, color = 'red')
	plt.plot(X_train, r.predict(X_train), color = 'blue')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()
#1er modele enginesize
x1 = df.iloc[:,[17]].values
y1 = df.iloc[:,26].values
regression(x1,y1,"la variance du prix en fonction de la dimension du vehicule","taille ")

#2eme modele carwidth
x2 = df.iloc[:,[12]].values
regression(x2,y1,"la variance du prix en fonction de la largeur du vehicule","largeur")

#3eme modele carlength
x3 = df.iloc[:,[11]].values
regression(x3,y1,"la variance du prix en fonction de la taille du vehicule","taille ")

#4eme modele horsepower
x4 = df.iloc[:,[22]].values
regression(x4,y1,"la variance du prix en fonction de la puissance du moteur","puissance du moteur")

#5eme modele curbweight
x5 = df.iloc[:,[14]].values
regression(x5,y1,"la variance du prix en fonction du poid à vide du vehicule","taille ")

#6eme modele
#quelque modification
df["cylindernumber"].value_counts()
df["cylindernumber"] = df["cylindernumber"].replace(["two","three"],"four")
df["cylindernumber"] = df["cylindernumber"].replace("twelve","eight")
df["drivewheel"].value_counts()
PV("cylindernumber").plot(kind="bar")

x6 =  pd.get_dummies(df["cylindernumber"])
x6 = x6.drop(columns="eight")
X_train, X_test, y_train, y_test = train_test_split(x6, y1, test_size = 0.20, random_state = 0)
r = LinearRegression()
r.fit(X_train, y_train)
y_pred = r.predict(X_test)
print('R2 score: %.2f' % r2_score(y_test,y_pred))

df.isnull().sum()



from sklearn.ensemble import RandomForestClassifier

df["cylindernumber"] = df["cylindernumber"].replace({"four":0,"five":1,"six":2,"eight":3},regex=True)
df["aspiration"] = df["aspiration"].replace({"std":0,"turbo":1})
df["drivewheel"] = df["drivewheel"].replace({"rwd":0,"fwd":1})
feature_cols = ["cylindernumber","aspiration","drivewheel"]
X = df[feature_cols]
Y = df["price_categorie"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf=RandomForestClassifier(n_estimators=30)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_,index=feature_cols).sort_values(ascending=False)
sns.barplot(x=feature_imp, y=feature_imp.index)

print(clf.predict([[3,1,0]])[0])


cm = confusion_matrix(y_test, y_pred)

def vis_confusion_matrix(cm):
     # name  of classes
    class_names=[0,1,2]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

vis_confusion_matrix(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap="Blues")
