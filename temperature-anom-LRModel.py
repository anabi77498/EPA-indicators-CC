import pandas as pd
import numpy as np

#Libraries for Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Manipulating global temperature data such that it gives us the last 70 years 
#where the trend was the most catastrophic

df5 = pd.read_csv("GlobalTemperatures.csv")
globtemps2 = df5
globtemps2['Year'] = 'NaN'
globtemps2['Year (int)'] = 'NaN'

for i in range(len(df5)):
    temp = df5['dt'].iloc[i][:4]
    globtemps2['Year'].iloc[i] = df5['dt'].iloc[i][:4]
    globtemps2['Year (int)'][i] = int(temp)


globtempsmodeldf = globtemps2.iloc[2400:]
globtempsmodeldf.head()

#seperate dataset that gives us a year to year overview of temperature recorded at the Dec of the year
newdf5 = df5.loc[df5['dt'] == '1965-12-01']
newdf5['Year'] = '1965'
newdf5['Year (int)'] = 1965

for i in range (1,53):
    newdf5 = newdf5.append((df5.loc[df5['dt'] == str(1965 + i)+'-12-01']),ignore_index = True)
    newdf5["Year"][i] = str(1965 + i)
    newdf5['Year (int)'][i] = 1965 + i

globtemps = newdf5
globtemps.head(55)

#transorming change in time to applicable dates

globtemps['dt'] = pd.to_datetime(globtemps['dt'])
globtemps.set_index('dt',inplace=True)
globtemps.index

#Linear Regression as an algorthmic predictor for Global Land Average Temperature (celsius)
def LinearReg(features,target,size,state,accuracy):
    xtrain,xtest,ytrain,ytest = train_test_split(features,target,test_size = size,random_state = state)
    model = LinearRegression()
    model.fit(xtrain,ytrain)
    prediction = model.predict(xtest)
    if accuracy == True:
        #accuracy
        modelscore = model.score(xtest,ytest)
        #DataFrame that compares true vs predicted
        modelframe = pd.DataFrame({'True Land Temperature':ytest, 
                                   'Model-predicted Land Temperature':prediction, 
                                   'Year':xtest['Year (int)']})
        print('Model Accuracy: ' + str(modelscore))
        return model,prediction,modelframe
    else:
        return model,prediction

#features of interest
tempfeatures = globtemps[['Year (int)','LandAndOceanAverageTemperature',
                          'LandAndOceanAverageTemperatureUncertainty']]
#target
temptarget = globtemps['LandAverageTemperature']

gtmodel, gtprediction, gtmodelframe = LinearReg(tempfeatures,temptarget,.4,42,True)

#DataFrame that compares true vs predicted
gtmodelframe.head()

# LinearRegression Equation
B1 = gtmodel.coef_[0]
Year = 2008
B2 = gtmodel.coef_[1]
LandAndOceanAverageTemperature = globtemps['LandAndOceanAverageTemperature'].mean()
B3 = gtmodel.coef_[2]
LandAndOceanAverageTemperatureUncertainty = globtemps['LandAndOceanAverageTemperatureUncertainty'].mean()
k = gtmodel.intercept_

# Uses LinearRegression and sklearn predict function to see future trends
def AverageTemperature(Year, LOAvgTemp = globtemps['LandAndOceanAverageTemperature'].mean() ,LOAvgTempU = globtemps['LandAndOceanAverageTemperatureUncertainty'].mean()):
    return round((gtmodel.predict([[Year,LOAvgTemp, LOAvgTempU]])[0]), 4)

def YearOfTemperature(Temp):
    LandAndOceanAverageTemperature = globtemps['LandAndOceanAverageTemperature'].mean()
    LandAndOceanAverageTemperatureUncertainty = globtemps['LandAndOceanAverageTemperatureUncertainty'].mean()
    return round(((Temp - gtmodel.coef_[1]*LandAndOceanAverageTemperature - gtmodel.coef_[2]*LandAndOceanAverageTemperatureUncertainty - gtmodel.intercept_)/gtmodel.coef_[0]))
    

Temp = round((B1*Year + B2*LandAndOceanAverageTemperature + B3*LandAndOceanAverageTemperatureUncertainty + k),6)
Year = round((Temp - B2*LandAndOceanAverageTemperature - B3*LandAndOceanAverageTemperatureUncertainty - k)/B1)
print(Temp,Year)