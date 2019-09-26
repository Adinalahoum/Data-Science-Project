import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

df = pd.read_csv("new.csv")
df = df.drop(columns="Unnamed: 0")

#Split data in features and target variable 
feature_cols = ["Credit_History","Self_Employed"]
X = df[feature_cols]
Y = df["Loan_Status"]

#Splitting the Data (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 

#Building Decision Tree Model 

#1)Create Decision Tree classifier object 
clf = DecisionTreeClassifier()
#2)Train Decision Tree Classifier 
clf = clf.fit(X_train,y_train)
#3)Predict the response for the test dataset 
y_pred = clf.predict(X_test)


#Accuracy of the model 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Visualization
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('resultat.png')
Image(graph.create_png())












