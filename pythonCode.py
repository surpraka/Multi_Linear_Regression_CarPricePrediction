# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:22:50 2020

@author: surpraka
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Data Preprocessing
car = pd.read_csv(r"C:\Users\surpraka\Desktop\MachineLearning\UpGrad\LR\Assignment\CarPrice_Assignment.csv")
car['CarName'] = car['CarName'].apply(lambda X: X.split(" ")[0])
car.drop('car_ID',axis=1,inplace=True)
car.CarName = car.CarName.str.replace('maxda','mazda').replace('vw','volkswagen').replace('toyouta','toyota').replace('Nissan','nissan').replace('vokswagen','volkswagen')

#CarBody
body = pd.get_dummies(car['carbody'],drop_first=True)
car = pd.concat([car,body],axis=1)
car.drop(['carbody'],axis=1,inplace=True)


#CarName
name = pd.get_dummies(car['CarName'],drop_first=True)
car = pd.concat([car,name],axis=1)
car.drop(['CarName'],axis=1,inplace=True)

#Fuel
fuel = pd.get_dummies(car['fueltype'],drop_first=True)
car = pd.concat([car,fuel],axis=1)
car.drop(['fueltype'],axis=1,inplace=True)

# Categorical Data Conversion
car['aspiration'] = car['aspiration'].map({'std':0,'turbo':1})
car['doornumber'] = car['doornumber'].map({'two':0,'four':1})
car['drivewheel'] = car['drivewheel'].map({'rwd':0,'fwd':1,'4wd':2})
car['enginelocation'] = car['enginelocation'].map({'front':0,'rear':1})

engineTypes = car['enginetype'].astype('category').cat.categories.tolist()
replace_engineType = {'enginetype' : {k: v for k,v in zip(engineTypes,list(range(1,len(engineTypes)+1)))}}
car.replace(replace_engineType,inplace=True)

le = LabelEncoder()
car['cylindernumber'] = le.fit_transform(car['cylindernumber'])

fuel = car['fuelsystem'].astype('category').cat.categories.tolist()
replace_fuel = {'fuelsystem' : {k: v for k,v in zip(fuel,list(range(1,len(fuel)+1)))}}
car.replace(replace_fuel,inplace=True)

# Split the data
df_train, df_test = train_test_split(car, train_size = 0.7, test_size = 0.3, random_state = 100)


num_columns = ['wheelbase','carlength','carwidth','carheight','curbweight',
               'enginesize','boreratio','stroke','compressionratio','horsepower'
               ,'peakrpm','citympg','highwaympg','price']

scaler = MinMaxScaler()
df_train[num_columns] = scaler.fit_transform(df_train[num_columns])

# Data Visualization
# ------------------
HeatMap for correl
plt.figure(figsize=[32,32])
sb.heatmap(df_train.corr(), annot=True,cmap='YlGnBu')
plt.show()

#Pairplot b/w price and enginesize

plt.figure(figsize=[10,10])
plt.scatter(car['enginesize'],car['price'])
plt.title('Engine Size')
plt.show()

plt.figure(figsize=[10,10])
plt.scatter(car['curbweight'],car['price'])
plt.title('Curb Weight')
plt.show()

plt.figure(figsize=[10,10])
plt.scatter(car['horsepower'],car['price'])
plt.title('HorsePower')
plt.show()

plt.figure(figsize=[10,10])
plt.scatter(car['carwidth'],car['price'])
plt.title('Car Width')
plt.show()

plt.figure(figsize=[10,10])
plt.scatter(car['carlength'],car['price'])
plt.title('Car Length')
plt.show()


# Model 1
# -------
y_train = df_train.pop('price')
X_train = df_train


lm = LinearRegression()
lm.fit(X_train,y_train)

# Automatic Feature Selection
rfe = RFE(lm,20)
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

X_train_lm = X_train[col]
X_train_lm = sm.add_constant(X_train_lm)
lr = sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())

# #---- VIF Score 1-------
vif = pd.DataFrame()
vif['features'] = X_train_lm.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm.values,i) for i in range(X_train_lm.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# # Model 2
# # -------
X_train_lm2 = X_train_lm.drop('compressionratio',1,)
X_train_lm2 = sm.add_constant(X_train_lm2)
lr2 = sm.OLS(y_train,X_train_lm2).fit()
print(lr2.summary())

# #---- VIF Score 2-------
vif = pd.DataFrame()
vif['features'] = X_train_lm2.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm2.values,i) for i in range(X_train_lm2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 3
# # -------

X_train_lm3 = X_train_lm2.drop('wheelbase',1,)
X_train_lm3 = sm.add_constant(X_train_lm3)
lr3 = sm.OLS(y_train,X_train_lm3).fit()
print(lr3.summary())

# #---- VIF Score 3-------
vif = pd.DataFrame()
vif['features'] = X_train_lm3.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm3.values,i) for i in range(X_train_lm3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# # Model 4
# # -------

X_train_lm4 = X_train_lm3.drop('carlength',1,)
X_train_lm4 = sm.add_constant(X_train_lm4)
lr4 = sm.OLS(y_train,X_train_lm4).fit()
print(lr4.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm4.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm4.values,i) for i in range(X_train_lm4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 5
# # -------

X_train_lm5 = X_train_lm4.drop('carheight',1,)
X_train_lm5 = sm.add_constant(X_train_lm5)
lr5 = sm.OLS(y_train,X_train_lm5).fit()
print(lr5.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm5.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm5.values,i) for i in range(X_train_lm5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# # Model 6
# # -------

X_train_lm6 = X_train_lm5.drop('enginesize',1,)
X_train_lm6 = sm.add_constant(X_train_lm6)
lr6 = sm.OLS(y_train,X_train_lm6).fit()
print(lr6.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm6.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm6.values,i) for i in range(X_train_lm6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 7
# # -------

X_train_lm7 = X_train_lm6.drop('carwidth',1,)
X_train_lm7 = sm.add_constant(X_train_lm7)
lr7 = sm.OLS(y_train,X_train_lm7).fit()
print(lr7.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm7.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm7.values,i) for i in range(X_train_lm7.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 8
# # -------

X_train_lm8 = X_train_lm7.drop('sedan',1,)
X_train_lm8 = sm.add_constant(X_train_lm8)
lr8 = sm.OLS(y_train,X_train_lm8).fit()
print(lr8.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm8.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm8.values,i) for i in range(X_train_lm8.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# # Model 9
# # -------

X_train_lm9 = X_train_lm8.drop('hardtop',1,)
X_train_lm9 = sm.add_constant(X_train_lm9)
lr9 = sm.OLS(y_train,X_train_lm9).fit()
print(lr9.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm9.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm9.values,i) for i in range(X_train_lm9.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)


# # Model 10
# # -------

X_train_lm10 = X_train_lm9.drop('saab',1,)
X_train_lm10 = sm.add_constant(X_train_lm10)
lr10 = sm.OLS(y_train,X_train_lm10).fit()
print(lr10.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm10.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm10.values,i) for i in range(X_train_lm10.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 10
# # -------

X_train_lm11 = X_train_lm10.drop('hatchback',1,)
X_train_lm11 = sm.add_constant(X_train_lm11)
lr11 = sm.OLS(y_train,X_train_lm11).fit()
print(lr11.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm11.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm11.values,i) for i in range(X_train_lm11.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# # Model 10
# # -------

X_train_lm12 = X_train_lm11.drop('gas',1,)
X_train_lm12 = sm.add_constant(X_train_lm12)
lr12 = sm.OLS(y_train,X_train_lm12).fit()
print(lr12.summary())

# #---- VIF Score 4-------
vif = pd.DataFrame()
vif['features'] = X_train_lm12.columns
vif['VIF'] = [variance_inflation_factor(X_train_lm12.values,i) for i in range(X_train_lm12.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF',ascending = False)
print(vif)

# #---- Residual Analysis -------
y_train_price = lr12.predict(X_train_lm12)

fig = plt.figure()
sb.distplot((y_train - y_train_price),bins = 20)
fig.suptitle('Error Terms',fontsize = 20)
plt.xlabel('Errors', fontsize = 18)

# --------------------------------------- Test Data -------------------------------------------------------------------------------

num_columns = ['wheelbase','carlength','carwidth','carheight','curbweight',
               'enginesize','boreratio','stroke','compressionratio','horsepower'
               ,'peakrpm','citympg','highwaympg','price']
scaler = MinMaxScaler()
df_test[num_columns] = scaler.fit_transform(df_test[num_columns])

y_test = df_test.pop('price')
X_test = df_test

X_test_lm = X_test[col]
X_test_lm = sm.add_constant(X_test_lm)
X_test_lm = X_test_lm.drop(['gas','hatchback','saab','hardtop','sedan','carwidth','enginesize','carheight','carlength','wheelbase','compressionratio'],axis=1)
y_test_price = lr12.predict(X_test_lm)

fig = plt.figure()
plt.scatter(y_test,y_test_price)
fig.suptitle('y_train vs y_train_price', fontsize=20)              
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_pred', fontsize=16)   

print(r2_score(y_test, y_test_price))