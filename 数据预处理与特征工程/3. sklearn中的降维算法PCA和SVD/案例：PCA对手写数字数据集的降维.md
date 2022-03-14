# 案例：PCA对手写数字数据集的降维

手写数字数据集

```python
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(r"F:\sklearn\03数据预处理和特征工程\digit_recognizor.csv")
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
print(X.shape)

''' 画累计方差贡献率曲线，找最佳降维后维度的范围'''
pca_line = PCA().fit(X)
plt.figure(figsize=[20, 5])
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()

'''降维后维度的学习曲线，继续缩小最佳维度的范围'''
score = []
for i in range(1, 101, 10):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10, random_state=0)
                           , X_dr, y, cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 101, 10), score)
plt.show()

'''细化学习曲线，找出降维后的最佳维度'''
score = []
for i in range(10, 25):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10, random_state=0), X_dr, y, cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(10, 25), score)
plt.show()

X_dr = PCA(23).fit_transform(X)

res = cross_val_score(RFC(n_estimators=100, random_state=0), X_dr, y, cv=5).mean()
print(res)
```