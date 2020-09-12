# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 22:14:26 2020

@author: Bhavana M
"""
#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt



#reading the data from files
data=pd.read_csv("advertising.csv")
data.head()


#visualising the data set
fig , axes = plt.subplots(1,3,sharey = True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axes[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axes[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axes[2])

#transforming the data
#creating x and y for linear regreaion
feature_cols=['TV']
X = data[feature_cols]
y = data.Sales


#importing linear regression algo for simple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


print(lr.intercept_)
print(lr.coef_)

#create a dataframe with max and min value of the table
x_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
x_new.head()


preds=lr.predict(x_new)
preds


data.plot(kind='scatter',x='TV',y='Sales')
plt.plot(x_new,preds,c='red',linewidth=3)

import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data= data).fit()


lm.conf_int()

#finding the probability values
lm.pvalues

#finding the R-squared values
lm.rsquared


#multi linear regression

feature_cols=['TV','Radio','Newspaper']
X = data[feature_cols]
y = data.Sales

lr=LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)


lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()



lm.pvalues
lm.rsquared

lm.summary()


