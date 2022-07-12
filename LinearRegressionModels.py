#%%
# Simple Linear Regression

import pandas as pd
df = pd.read_csv("Advertising.csv")
df = df.iloc[:,1:len(df)]
#df.head()
#df.info()

import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"

#sns.jointplot(x= "TV",y= "sales",data=df,kind="reg")

from sklearn.linear_model import LinearRegression

X = df[["TV"]]
Y = df[["sales"]]

linear_reg = LinearRegression()
linear_reg.fit(X,Y)

print("intercept:",linear_reg.intercept_)
print("coef:",linear_reg.coef_)
print("score:",linear_reg.score(X,Y))


#%%
# Prediction

import matplotlib.pyplot as plt
g = sns.regplot(df["TV"],df["sales"],ci=None,scatter_kws=({'color':'r','s':9}))
g.set_title("Model Formula: Sales = 7.03 + TV * 0.047")
g.set_ylabel("Sales Value")
g.set_xlabel("TV Expenses")
plt.xlim(-10,310)
plt.ylim(bottom=0)

# linear_reg.intercept_ + linear_reg.coef_*165 = linear_reg.predict([[165]])

#%%
# Residuals
linear_reg.predict(X)[0:6]

real_y = Y[0:10]
predicted_y = pd.DataFrame(linear_reg.predict(X)[0:10])
rpFrame = pd.concat([real_y,predicted_y],axis=1)
rpFrame.columns = ["real_y","predicted_y"]

rpFrame["residuals"] = rpFrame["real_y"] - rpFrame["predicted_y"]
rpFrame["residualsSquare"] = rpFrame["residuals"]**2

rpFrame["residualsSquare"].mean()



#%%
# Multiple Linear Regression
X2 = df.drop("sales",axis=1)
Y2 = df[["sales"]]

# Statsmodels 
import statsmodels.api as sm

lm = sm.OLS(Y2,X2)
MLRModel = lm.fit()
MLRModel.summary()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X2,Y2)
print("intercept:",lr.intercept_)
print("coef:",lr.coef_)

# lr.intercept_ + TV*lr.coef_[0][0] + radio*lr.coef_[0][1] + newspaper*lr.coef_[0][2]
# 2.938         + TV*0.045       + radio*0.188       + newspaper*-0.001

# 30 TV + 10 Radio + 40 Newspaper?
print("answer:",lr.intercept_ + 30*lr.coef_[0][0] + 10*lr.coef_[0][1] + 40*lr.coef_[0][2])
print("predict:",lr.predict([[30,10,40]]))

# Mean Squared Error
from sklearn.metrics import mean_squared_error
import numpy as np

print("MSE:",mean_squared_error(Y2, lr.predict(X2)))
print("RMSE:",np.sqrt(mean_squared_error(Y2, lr.predict(X2))))

#%%
# MLR test
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X2, Y2, 
                                                 test_size=0.2, random_state=(99))

lr = LinearRegression()
lr.fit(X_train,Y_train)

# Train Error
print("RMSE Train:", np.sqrt(mean_squared_error(Y_train, 
                                 lr.predict(X_train))))

# Test Error
print("RMSE Test:", np.sqrt(mean_squared_error(Y_test, 
                                 lr.predict(X_test))))

# Cross_Val Score
from sklearn.model_selection import cross_val_score

# Cross Validation MSE
print("Cross Validation MSE: ",np.mean(-cross_val_score(lr, X_train, Y_train, cv=10, scoring="neg_mean_squared_error")))

# Cross Validation RMSE
print("Cross Validation RMSE: ",np.sqrt(np.mean(-cross_val_score(lr, X_train, Y_train, cv=10, scoring="neg_mean_squared_error"))))




#%%











#%%