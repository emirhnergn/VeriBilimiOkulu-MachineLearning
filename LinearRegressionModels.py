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
# Ridge Regression
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv("Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    random_state=42)

ridge_model = Ridge(alpha=5)
ridge_model.fit(x_train,y_train)

print("intercept:",ridge_model.intercept_)
print("coef:",ridge_model.coef_)

lambdas = 10**np.linspace(10,-2,100)*0.5
ridge_model2 = Ridge()
coefficients = []

for i in lambdas:
    ridge_model2.set_params(alpha = i)
    ridge_model2.fit(x_train,y_train)
    coefficients.append(ridge_model2.coef_)

ax = plt.gca()
ax.plot(lambdas, coefficients)
ax.set_xscale("log")

#%%
# Ridge Regression Prediction

ridge_model3 = Ridge()
ridge_model3.fit(x_train, y_train)
y_pred = ridge_model3.predict(x_train)

# train error
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred))
#RMSE_train
print("Cross Validation MSE: ",np.sqrt(np.mean(-cross_val_score(ridge_model3, x_train, y_train, cv=10, scoring="neg_mean_squared_error"))))

# test error
y_pred = ridge_model3.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
#RMSE_test


#%%
# Ridge Regression Model Tuning
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ridge_model4 = Ridge()
ridge_model4.fit(x_train, y_train)
y_pred = ridge_model4.predict(x_train)
#np.sqrt(mean_squared_error(y_train, y_pred))

lambdas1 = np.random.randint(0, 1000, 100)
lambdas2 = 10**np.linspace(10,-2,100)*0.5

ridgecv = RidgeCV(alphas=lambdas2, scoring= "neg_mean_squared_error", 
                  cv = 10)
ridgecv_pipeline = make_pipeline(StandardScaler(), ridgecv)

ridgecv_pipeline.fit(x_train, y_train)
ridgecv.alpha_

# Final Ridge Model
ridge_tuned = Ridge(alpha = ridgecv.alpha_).fit(x_train, y_train)
y_pred = ridge_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned Ridge RMSE: ", RMSE_test)


#%%
# Lasso Regression
from sklearn.linear_model import Lasso, LassoCV

df = pd.read_csv("Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    random_state=42)

lasso_model = Lasso(max_iter = 5000)
lasso_model.fit(x_train, y_train)
print("intercept:",lasso_model.intercept_)
print("coef:",lasso_model.coef_)

lasso_model2 = Lasso(max_iter = 50000)
coefs = []
alphas1 = np.random.randint(0, 100000, 100)
alphas2 = 10**np.linspace(10,-2,100)*0.5

for a in alphas2:
    lasso_model2.set_params(alpha = a)
    lasso_model2.fit(x_train, y_train)
    coefs.append(lasso_model2.coef_)

ax = plt.gca()
ax.plot(alphas2, coefs)
ax.set_xscale("log")



#%%
# Lasso Regression Prediction
y_pred = lasso_model.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Lasso RMSE: ", RMSE_test)
print("Lasso R2 Score: ", r2_score(y_test,y_pred))


#%%
# Lasso Regression Model Tuning
alphas1 = np.random.randint(0, 100000, 100)
alphas2 = 10**np.linspace(10,-2,100)*0.5

lasso_cv_model = LassoCV(alphas=alphas2, cv = 10, max_iter=100000)
lasso_cv_model.fit(x_train,y_train)
lasso_cv_model.alpha_

lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(x_train, y_train)
y_pred = lasso_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned Lasso RMSE: ", RMSE_test)
print("Tuned Lasso R2 Score: ", r2_score(y_test,y_pred))

pd.Series(lasso_tuned.coef_, index = x_train.columns)

#%%
# ElasticNet
from sklearn.linear_model import ElasticNet, ElasticNetCV

df = pd.read_csv("Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    random_state=42)

enet_model = ElasticNet(max_iter = 5000)
enet_model.fit(x_train, y_train)
print("intercept:",enet_model.intercept_)
print("coef:",enet_model.coef_)

y_pred = enet_model.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("ElasticNet RMSE: ", RMSE_test)
print("ElasticNet R2 Score: ", r2_score(y_test,y_pred))


#%%
# ElasticNet Model Tuning
alphas1 = np.random.randint(0, 100000, 100)
alphas2 = 10**np.linspace(10,-2,100)*0.5

enet_cv_model = ElasticNetCV(alphas=alphas2, cv = 10, max_iter=100000)
enet_cv_model.fit(x_train,y_train)
enet_cv_model.alpha_

enet_tuned = ElasticNet().set_params(alpha = enet_cv_model.alpha_)
enet_tuned.fit(x_train, y_train)
y_pred = enet_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned ElasticNet RMSE: ", RMSE_test)
print("Tuned ElasticNet R2 Score: ", r2_score(y_test,y_pred))

pd.Series(enet_tuned.coef_, index = x_train.columns)


#%%