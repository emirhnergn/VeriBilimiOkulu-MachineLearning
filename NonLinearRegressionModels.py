#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from warnings import filterwarnings
filterwarnings("always")
filterwarnings("ignore")


df = pd.read_csv("Hitters.csv")
df = df.dropna()

dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    random_state=42)

# KNN

knn_model= KNeighborsRegressor()
knn_model.fit(x_train, y_train)
y_pred = knn_model.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("KNN RMSE: ", RMSE_test)
print("KNN R2 Score: ", r2_score(y_test,y_pred))


#%%
# KNN Tuning
RMSE = []

for k in range(1,11):
    knn_model= KNeighborsRegressor(n_neighbors = k)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    RMSE.append(rmse)
    print("k=",k,"RMSE: ", rmse)

# GridSearchCV

knn_params = {
    "n_neighbors":np.arange(1,30,1)
    }
knn_model = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10)
knn_cv_model.fit(x_train,y_train)

knn_cv_model.best_params_

# KNN Tuned
knn_model_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_model_tuned.fit(x_train,y_train)
y_pred = knn_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned KNN RMSE:", RMSE_test)


#%%
# Support Vector Regression
svr_model = SVR(kernel="linear")
svr_model.fit(x_train,y_train)
y_pred = svr_model.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("SVR RMSE: ", RMSE_test)
print("SVR R2 Score: ", r2_score(y_test,y_pred))
#%%
# SVR Tuning
svr_model = SVR(kernel="linear")
svr_params = {
    "C" : [0.1,0.5]
    }
svr_cv_model = GridSearchCV(svr_model, svr_params,verbose=True,cv=5,n_jobs=-1)
svr_cv_model.fit(x_train, y_train)
svr_cv_model.best_params_

svr_model_tuned = SVR(kernel="linear", C=svr_cv_model.best_params_["C"])
svr_model_tuned.fit(x_train,y_train)
y_pred = svr_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned SVR RMSE:", RMSE_test)


#%%
# ANN 
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
scaler.fit(x_test)
x_test_scaled = scaler.transform(x_test)

mlp_model = MLPRegressor()
mlp_model.fit(x_train_scaled,y_train)
y_pred = mlp_model.predict(x_test_scaled)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("ANN RMSE: ", RMSE_test)
print("ANN R2 Score: ", r2_score(y_test,y_pred))
# ANN Tuning

mlp_params = {
    "alpha" : [0.1, 0.01, 0.02, 0.001, 0.0001],
    "hidden_layer_sizes" : [(5,5), (10,20), (100,100)]
    }

mlp_model = MLPRegressor(max_iter = 500)
mlp_cv_model = GridSearchCV(mlp_model, mlp_params,
                            verbose=True,cv=10,n_jobs=-1)
mlp_cv_model.fit(x_train_scaled, y_train)
print("ANN Best Params:",mlp_cv_model.best_params_)

mlp_model_tuned = MLPRegressor(alpha=mlp_cv_model.best_params_["alpha"],
                               hidden_layer_sizes=mlp_cv_model.best_params_["hidden_layer_sizes"])
mlp_model_tuned.fit(x_train_scaled,y_train)
y_pred = mlp_model_tuned.predict(x_test_scaled)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned ANN RMSE:", RMSE_test)

#%%
# CART
x_train_hits = pd.DataFrame(x_train["Hits"])
x_test_hits = pd.DataFrame(x_test["Hits"])

cart_model = DecisionTreeRegressor(max_leaf_nodes = 10)
cart_model.fit(x_train_hits, y_train)

x_grid = np.arange(min(np.array(x_train_hits)),max(np.array(x_train_hits)),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x_train_hits, y_train, color="red")

plt.plot(x_grid, cart_model.predict(x_grid), color="blue")

plt.title("CART Regressor")
plt.xlabel("Hits")
plt.ylabel("Salary")

#%%
# CART Prediction (1 Variable)
cart_model = DecisionTreeRegressor(max_leaf_nodes = 10)
cart_model.fit(x_train_hits, y_train)

y_pred = cart_model.predict(x_test_hits)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("CART (1) RMSE: ", RMSE_test)
print("CART (1) R2 Score: ", r2_score(y_test,y_pred))
# CART Prediction (Multiple Variable)
cart_model = DecisionTreeRegressor(max_leaf_nodes = 10)
cart_model.fit(x_train, y_train)

y_pred = cart_model.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("CART RMSE: ", RMSE_test)
print("CART R2 Score: ", r2_score(y_test,y_pred))

#%%
# CART Tuning
cart_params = {
    "max_depth" : [2,3,4,5,6,7,10,15,20],
    "max_leaf_nodes" : [10,15,20,30,40,45,50,55,70],
    "min_samples_split" : [2,5,10,20,30,35,40,45,50,55,100],
    }

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params,
                            verbose=True,cv=10,n_jobs=-1)
cart_cv_model.fit(x_train, y_train)
print("CART Best Params:",cart_cv_model.best_params_)

cart_model_tuned = DecisionTreeRegressor(max_leaf_nodes=cart_cv_model.best_params_["max_leaf_nodes"],
                               max_depth=cart_cv_model.best_params_["max_depth"],
                               min_samples_split=cart_cv_model.best_params_["min_samples_split"])
cart_model_tuned.fit(x_train,y_train)
y_pred = cart_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned CART RMSE:", RMSE_test)





#%%






#%%