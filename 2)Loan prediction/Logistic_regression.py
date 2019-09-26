    #Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#Import the new dataset after analyse
df = pd.read_csv("new.csv")
df = df.drop(columns="Unnamed: 0")
#Generic function for making a classification model and accessing performance:
#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome,n):
  #Fit the model:
  model.fit(data[predictors],data[outcome])

  #Make predictions on training set:
  predictions = model.predict(data[predictors])

  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(n)
  error = []
  for train, test in kf.split(data[predictor_var]):
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])

    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]

    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)

    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
  print(error)
  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])
#the first Logistic Regression Model

outcome_var = 'Loan_Status'
model = LogisticRegression(solver='lbfgs')
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var,10)

def vis_confusion_matrix(cm):
     # name  of classes
    class_names=[0,1]
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

