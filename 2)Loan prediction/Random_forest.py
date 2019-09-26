import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold   #For K-fold cross validation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
df = pd.read_csv("new.csv")
df = df.drop(columns="Unnamed: 0")

#Split data in features and target variable
feature_cols = ['Dependents','Credit_History','LoanAmount_log']
X = df[feature_cols]
Y = df["Loan_Status"]

#Splitting the Data (70% training and 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print cross validation score

#Check the feature importance score
feature_imp = pd.Series(clf.feature_importances_,index=feature_cols).sort_values(ascending=False)
print(feature_imp)

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

#Visualization
# Extract single tree
estimator = clf.estimators_[5]
# Export as dot file
dot_data = StringIO()
export_graphviz(estimator, out_file=dot_data,
                rounded = True, proportion = False,
                precision = 2, filled = True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('resultat2.png')







