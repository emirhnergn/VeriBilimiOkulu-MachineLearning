#%%
import seaborn as sns
df= sns.load_dataset('diamonds')
df = df.select_dtypes(include = ["float64", "int64"])
df = df.dropna()
df.head()

df_table = df["table"]

import plotly_express as px
import plotly.io as pio
pio.renderers.default = "svg"

fig = px.box(df_table)
fig.show()

Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3-Q1

bottom_line = Q1 - 1.5*IQR
top_line = Q3 + 1.5 *IQR

inconsistent_tf =  ((df_table < bottom_line) | (df_table > top_line)) 
#df_table[inconsistent_tf].index

#%%
# Delete
import pandas as pd

df_table = pd.DataFrame(df_table)
clear_df = df_table[~(inconsistent_tf)]
#%%
# Fill Median
import copy
df_table_median = copy.deepcopy(df_table) 
df_table_median[inconsistent_tf] = df_table_median.mean() 

#%%
# Rounding
df_table_round = copy.deepcopy(df_table) 
df_table_round[(df_table < bottom_line)] = bottom_line
df_table_round[(df_table > top_line)] = top_line

#%%
# Local Outlier Factor
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

LOF_df = copy.deepcopy(df)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(LOF_df)

df_scores = clf.negative_outlier_factor_
threshold_value =  np.sort(df_scores)[13]
LOF_inconsistent_tf = df_scores > threshold_value

new_df = LOF_df[df_scores > threshold_value]

#%%
# Local Outlier Factor + Rounding
round_value =  LOF_df[df_scores == threshold_value]
inconsistens = LOF_df[~LOF_inconsistent_tf]

res = inconsistens.to_records(index=False)
res[:] = round_value.to_records(index= False)

LOF_df[~LOF_inconsistent_tf] = pd.DataFrame(res, index = LOF_df[~LOF_inconsistent_tf].index)



#%%
# Lost Value 
import numpy as np
import pandas as pd
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df2 = pd.DataFrame(
    {
     "V1" : V1,
     "V2" : V2,
     "V3" : V3,
     })
#df2[df2.isnull().any(axis=1)]
#df2[df2.notnull().all(axis=1)]
#%%
# Delete Lost Value
df2.dropna(inplace=False)
df2.dropna(how="all")
df2.dropna(axis=1)
df2.dropna(axis=1,how="all")
df2.dropna(axis=1,how="all",inplace=False)
#%%
# Fill with Mean
df2["V1"].fillna(df2["V1"].mean(), inplace=False)
df2.apply(lambda x : x.fillna(x.mean()),axis=0)


#%%
# Visualize

#sns.heatmap(df2)

import missingno as msno
msno.bar(df2, figsize=(15,5))
msno.matrix(df2, figsize=(15,5))

#%%
import seaborn as sns
df3 = sns.load_dataset("planets")


msno.matrix(df3, figsize=(15,5))
msno.heatmap(df3,figsize=(15,5))

#%%
# Fill Values with Categorical Value
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])
df4 = pd.DataFrame(
    {
     "salary" : V1,
     "V2" : V2,
     "V3" : V3,
     "department" : V4,
     })

df4.groupby("department")["salary"].mean()
df4["salary"].fillna(df4.groupby("department")["salary"].transform("mean"))

#%%
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.NaN,"IK","IK","IK","IK","IK","IT","IT"],dtype="object")
df5 = pd.DataFrame(
    {
     "salary" : V1,
     "department" : V4,
     })

df5["department"].fillna(df5["department"].mode()[0])
df5["department"].fillna(method="bfill")
df5["department"].fillna(method="ffill")

#%%
# Fill with Prediction
import seaborn as sns
import missingno as msno
import numpy as np
import pandas as pd

df6 = sns.load_dataset('titanic')
df6 = df6.select_dtypes(include = ["float64", "int64"])
#print(df6.head())
df6.isnull().sum()

# KNN
from ycimpute.imputer import knnimput

var_names = list(df6)
n_df6 = np.array(df6)

knn6 = knnimput.KNN(k = 4).complete(n_df6)
knn6 = pd.DataFrame(knn6,columns = var_names)
#dff6.isnull().sum()

# RanndomForest
from ycimpute.imputer import iterforest
# Broken
#rf6 = iterforest.IterImput().complete(n_df6)
#rf6 = pd.DataFrame(rf6, columns = var_names)

# EM
from ycimpute.imputer import EM
em6 = EM().complete(n_df6)
em6 = pd.DataFrame(em6, columns = var_names)

#%%
# Data Standardization
V1 = np.array([1,3,6,5,7])
V2 = np.array([7,7,5,8,12])
V3 = np.array([6,12,5,6,14])
df7 = pd.DataFrame(
    {
     "V1" : V1,
     "V2" : V2,
     "V3" : V3,
     })
df7 = df7.astype(float)

# Standardization
from sklearn import preprocessing
preprocessing.scale(df7)

# Normalization
preprocessing.normalize(df7)

# Min-Max
preprocessing.minmax_scale(df7)
minmaxscaler = preprocessing.MinMaxScaler(feature_range=(10,20))
minmaxscaler.fit_transform(df7)
#%%
# Variable Transformation
df8 = sns.load_dataset('tips')
df8.head()

# 0-1 Transform
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
lbe.fit_transform(df8["sex"])
df8["new_sex"] = lbe.fit_transform(df8["sex"])

# Select 1 - Others 0 Transform
df8["new_day"] = np.where(df8["day"].str.contains("Sun"),1,0)


# Transform to Classes
lbe2 = LabelEncoder()
lbe2.fit_transform(df8["day"])

# One-Hot Encoder
df8_one_hot = pd.get_dummies(df8, columns=["sex"], prefix=["sex"])
df8_one_hot2 = pd.get_dummies(df8, columns=["day"], prefix=["day"])

#%%







#%%