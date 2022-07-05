import pandas as pd
import numpy as np

#Libraries for Kneighbors Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

Fooddf = pd.read_csv("Food_Production copy.csv")

#Manual Classification of livestock as an Animal or not (Be used as comparision)
Fooddf['Animal'] = False

for i in range(len(Fooddf)):
    if Fooddf['Animal Feed'].iloc[i] != 0:
        Fooddf['Animal'].iloc[i] = True
    
Fooddf.iloc[40] = Fooddf.iloc[42].fillna(Fooddf.iloc[41])
        
for i in Fooddf:
    if i != 'Food product':
        Fooddf[i] = Fooddf[i].fillna(Fooddf[i].iloc[:30].mean())
    
# Classification of whether an Agricultural Produce is an Animal or Not based on Environmental Correlations

def KNeighborsClassif(features,target,size,state,n_neighbors,accuracy):
    xtrain,xtest,ytrain,ytest = train_test_split(features,target,test_size = size,random_state=state)
    model = KNeighborsClassifier(n_neighbors)
    model.fit(xtrain,ytrain)
    if accuracy == True:
        testprediction = model.predict(xtest)
        trainprediction = model.predict(xtrain)

        testscore = accuracy_score(testprediction,ytest)
        trainscore = accuracy_score(trainprediction,ytrain)

        print("Train Accuracy Score for KNN model: " + str(trainscore))
        print("Test Accuracy Score for KNN model: " + str(testscore))
    return model

ClimateImplications = Fooddf[['Farm','Retail','Total_emissions', 'Eutrophying emissions per 1000kcal (gPO₄eq per 1000kcal)','Eutrophying emissions per kilogram (gPO₄eq per kilogram)','Freshwater withdrawals per 1000kcal (liters per 1000kcal)', 'Greenhouse gas emissions per 100g protein (kgCO₂eq per 100g protein)']]
features = ClimateImplications
target = Fooddf['Animal']

meatmodel = KNeighborsClassif(features,target,.25,53,3,True)