import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 1-1) counts the number of NAN rows for each feature
def count_nan_rows():
    print(f'sepal_length :  {df["sepal_length"].isna().sum()}')
    print(f'sepal_width  :  {df["sepal_width"].isna().sum()}')
    print(f'petal_length :  {df["petal_length"].isna().sum()}')
    print(f'petal_width  :  {df["petal_width"].isna().sum()}')
    print(f'target       :  {df["target"].isna().sum()}\n')


# 1-2) removes each row that contains the NAN feature
def drop_nan_value(data_frame):
    return data_frame.dropna()


def statistics_info():
    variances = df[features].var()
    print(df[features].describe())
    print('var', end="       ")
    for i in variances.values:
        print(format(i, ".6f"), end="     ")
    print('\n')


# Load dataset
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.read_csv("../iris.data", names=features + ['target'])

count_nan_rows()
df = drop_nan_value(df)
count_nan_rows()
# 2-1) label encoder for target feature
df['target'] = LabelEncoder().fit_transform(df.target.values)
statistics_info()
# 3-1) normalize features
df_numpy = StandardScaler().fit_transform(df[features])
df_numpy = np.append(df_numpy, df[['target']].to_numpy(), axis=1)
df = pd.DataFrame(df_numpy, columns=features + ['target'])
statistics_info()
# 4) PCA
x = df.loc[:, features].values
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
result_df = pd.concat([principal_df, df[['target']]], axis=1)
# 5) visualization
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig = plt.figure(figsize=(7, 7))
ax1.set_xlabel('PC1', fontsize=18)
ax1.set_ylabel('PC2', fontsize=18)
targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    is_equal = result_df['target'] == target
    ax1.scatter(result_df.loc[is_equal, 'PC1'],
                result_df.loc[is_equal, 'PC2'], c=color, s=50)
ax1.legend(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
ax1.grid()
boxplot = df.boxplot(column=features, ax=ax2)
plt.show()
