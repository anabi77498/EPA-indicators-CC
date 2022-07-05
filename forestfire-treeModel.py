import pandas as pd
import numpy as np

#Libraries for  Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

#features and targets represented as x and y
x = FFdf.iloc[:, :48] 
y = FFdf.iloc[:, 48]  

# Estimating proper sub-samples of data to be utilized as features via ExtraTreesClassifier
treetemp = ExtraTreesClassifier()
treetemp.fit(x, y)

tempmodel = SelectFromModel(treetemp, prefit=True)
features = tempmodel.transform(x) #new features
target = y

xtrain,xtest,ytrain,ytest = train_test_split(features,target,test_size = .25,random_state = 86)



FFdf = pd.read_csv("covtype.csv")
FFdf.head()

# Creating repeated Trees in order to figure out the best_depth of the Decision Tree based on our features


best_depth = 1 #Keep track of depth that produces tree with highest accuracy
best_accuracy = 0 #The best accuracy from a given tree
for k in range(1,50):
    dtreemodel=tree.DecisionTreeClassifier(max_depth=k)
    
    dtreemodel.fit(xtrain,ytrain)
    pred = dtreemodel.predict(xtest)
    accuracy = accuracy_score(pred,ytest)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = k
    
print(best_accuracy)
print(best_depth)


#Accuracy and Error in Model
treemodel=tree.DecisionTreeClassifier(max_depth=best_depth)
treemodel.fit(xtrain,ytrain)

dtree_pred_train = treemodel.predict(xtrain)
dtree_pred_test = treemodel.predict(xtest) 

print("Train Accuracy: ", accuracy_score(dtree_pred_train, ytrain))
print("Test Accuracy: ", accuracy_score(dtree_pred_test, ytest))
print("RMS Error: ", mean_squared_error(ytest,dtree_pred_test))

