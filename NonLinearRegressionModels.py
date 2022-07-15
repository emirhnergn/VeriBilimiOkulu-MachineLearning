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
# Random Forest
rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(x_train, y_train)

y_pred = rf_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Random Forest Tuning

rf_params = {
    "max_depth" : [5,8],
    "max_features" : [2,5],
    "n_estimators" : [200,500,2000],
    "min_samples_split": [2,10,100]
    }

rf_model = RandomForestRegressor()
rf_cv_model = GridSearchCV(rf_model, rf_params,
                            verbose=2,cv=10,n_jobs=-1)
rf_cv_model.fit(x_train, y_train)
print("RF Best Params:",rf_cv_model.best_params_)

rf_model_tuned = RandomForestRegressor(max_depth=rf_cv_model.best_params_["max_depth"],
                               max_features=rf_cv_model.best_params_["max_features"],
                               n_estimators=rf_cv_model.best_params_["n_estimators"],
                               min_samples_split = rf_cv_model.best_params_["min_samples_split"])
rf_model_tuned.fit(x_train,y_train)
y_pred = rf_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned RF RMSE:", RMSE_test)
#%%
# Variable Importance for rf_model
Importance = pd.DataFrame({"Importance": rf_model_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh",
                                              color = "r")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None

#%%
# Gradient Boosting Machines
gbm_model = GradientBoostingRegressor()
gbm_model.fit(x_train, y_train)

y_pred = gbm_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#Gradient Boosting Machines Tuning

gbm_params = {
    "max_depth" : [3,5,8],
    "learning_rate" : [0.001, 0.01, 0.1],
    "n_estimators" : [100,200,500],
    "subsample": [1, 0.5, 0.8],
    "loss" : ["absolute_error", "quantile"]
    }

gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model, gbm_params,
                            verbose=2,cv=10,n_jobs=-1)
gbm_cv_model.fit(x_train, y_train)
print("GBM Best Params:",gbm_cv_model.best_params_)

gbm_model_tuned = GradientBoostingRegressor(max_depth = gbm_cv_model.best_params_["max_depth"],
                               learning_rate = gbm_cv_model.best_params_["learning_rate"],
                               n_estimators = gbm_cv_model.best_params_["n_estimators"],
                               subsample = gbm_cv_model.best_params_["subsample"],
                               loss = gbm_cv_model.best_params_["loss"])
gbm_model_tuned.fit(x_train,y_train)
y_pred = gbm_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned GBM RMSE:", RMSE_test)
# Variable Importance for gbm_model
Importance = pd.DataFrame({"Importance": gbm_model_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh",
                                              color = "r")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None

#%%
# XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
print("Default XGB RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))

#XGBoost Tuning

xgb_params = {
    "max_depth" : [2,3,4,5,8],
    "learning_rate" : [0.1,0.01,0.5],
    "n_estimators" : [100,200,500],
    "colsample_bytree": [0.4, 0.7, 1]
    }

xgb_model = XGBRegressor()
xgb_cv_model = GridSearchCV(xgb_model, xgb_params,
                            verbose=2,cv=10,n_jobs=-1)
xgb_cv_model.fit(x_train, y_train)
print("XGB Best Params:",xgb_cv_model.best_params_)

xgb_model_tuned = XGBRegressor(max_depth = xgb_cv_model.best_params_["max_depth"],
                               learning_rate = xgb_cv_model.best_params_["learning_rate"],
                               n_estimators = xgb_cv_model.best_params_["n_estimators"],
                               colsample_bytree = xgb_cv_model.best_params_["colsample_bytree"])
xgb_model_tuned.fit(x_train,y_train)
y_pred = xgb_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned XGB RMSE:", RMSE_test)
# Variable Importance for xgb_model
Importance = pd.DataFrame({"Importance": xgb_model_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh",
                                              color = "r")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
#%%
# LightGBM
from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor()
lgbm_model.fit(x_train, y_train)

y_pred = lgbm_model.predict(x_test)
print("Default LightGBM RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))

#LightGBM Tuning

lgbm_params = {
    "max_depth" : [1,2,3,4,5,6,7,8,9,10],
    "learning_rate" : [0.1,0.01,0.5,1],
    "n_estimators" : [20,40,100,200,500,1000],
    "colsample_bytree": [0.4, 0.7, 1]
    }

lgbm_model = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params,
                            verbose=1,cv=10,n_jobs=-1)
lgbm_cv_model.fit(x_train, y_train)
print("LightGBM Best Params:",lgbm_cv_model.best_params_)

lgbm_model_tuned = LGBMRegressor(max_depth = lgbm_cv_model.best_params_["max_depth"],
                               learning_rate = lgbm_cv_model.best_params_["learning_rate"],
                               n_estimators = lgbm_cv_model.best_params_["n_estimators"],
                               colsample_bytree = lgbm_cv_model.best_params_["colsample_bytree"])
lgbm_model_tuned.fit(x_train,y_train)
y_pred = lgbm_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned LightGBM RMSE:", RMSE_test)
# Variable Importance for lgbm_model
Importance = pd.DataFrame({"Importance": lgbm_model_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh",
                                              color = "r")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None
#%%
# CatBoost
from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor()
catboost_model.fit(x_train, y_train)

y_pred = catboost_model.predict(x_test)
print("Default CatBoost RMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))

#CatBoost Tuning

catboost_params = {
    "depth" : [3,6,8],
    "learning_rate" : [0.01, 0.1],
    "iterations" : [200, 500, 1000]
    }

catboost_model = CatBoostRegressor()
catboost_cv_model = GridSearchCV(catboost_model, catboost_params,
                            verbose=1,cv=5,n_jobs=-1)
catboost_cv_model.fit(x_train, y_train)
print("CatBoost Best Params:",catboost_cv_model.best_params_)

catboost_model_tuned = CatBoostRegressor(depth = catboost_cv_model.best_params_["depth"],
                               learning_rate = catboost_cv_model.best_params_["learning_rate"],
                               iterations = catboost_cv_model.best_params_["iterations"])
catboost_model_tuned.fit(x_train,y_train)
y_pred = catboost_model_tuned.predict(x_test)
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred))
print("Tuned CatBoost RMSE:", RMSE_test)
# Variable Importance for catboost_model
Importance = pd.DataFrame({"Importance": catboost_model_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh",
                                              color = "r")
plt.xlabel("Variable Importance")
plt.gca().legend_ = None

#%%
# Automation
def compML(df, y, alg):
    # train-test split
    y = df[y]
    x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
    x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                        random_state=42)
    
    # modelling
    model = alg().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    print(alg.__name__,"RMSE :", RMSE)
    return RMSE

model = [
    LGBMRegressor,
    XGBRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    DecisionTreeRegressor,
    MLPRegressor,
    KNeighborsRegressor,
    SVR
    ]

for regressor in model:
    compML(df, "Salary", regressor)
    

#%%