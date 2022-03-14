# sklearn中的降维算法PCA和SVD

sklearn中的降维算法

sklearn中降维算法都被包括在模块decomposition中，这个模块本质是一个矩阵分解模块。

PCA与SVD

在降维过程中，我们会减少特征的数量，这意味着删除数据，数据量变少则表示模型可以获取的信息会变少，模型的表现可能会因此受影响。同时，在高维数据中，必然有一些特征是不带有有效的信息的（比如噪音），或者有一些特征带有的信息和其他一些特征是重复的（比如一些特征可能会线性相关）。我们希望能够找出一种办法来帮助我们衡量特征上所带的信息量，让我们在降维的过程中，能够即减少特征的数量，又保留大部分有效信息——将那些带有重复信息的特征合并，并删除那些带无效信息的特征等等——逐渐创造出能够代表原特征矩阵大部分信息的，特征更少的，新特征矩阵。

上周的特征工程课中，我们提到过一种重要的特征选择方法：方差过滤。如果一个特征的方差很小，则意味着这个特征上很可能有大量取值都相同（比如90%都是1，只有10%是0，甚至100%是1），那这一个特征的取值对样本而言就没有区分度，这种特征就不带有有效信息。从方差的这种应用就可以推断出，**如果一个特征的方差很大，则说明这个特征上带有大量的信息**。因此，在降维中，PCA使用的信息量衡量指标，就是样本方差，又称可解释性方差，方差越大，特征所带的信息量越多。

PCA和SVD是两种不同的降维算法，但他们都遵从上面的过程来实现降维，只是两种算法中矩阵分解的方法不同，信息量的衡量指标不同罢了。**PCA使用方差作为信息量的衡量指标，并且特征值分解来找出空间V。**降维时，它会通过一系列数学的神秘操作（比如说，产生协方差矩阵 ）将特征矩阵X分解为以下三个矩阵，其中 和 是辅助的矩阵，Σ是一个对角矩阵（即除了对角线上有值，其他位置都是0的矩阵），其对角线上的元素就是方差。降维完成之后，PCA找到的每个新特征向量就叫做“主成分”，而被丢弃的特征向量被认为信息量很少，这些信息很可能就是噪音。

![Untitled](sklearn%E4%B8%AD%E7%9A%84%E9%99%8D%2081266/Untitled.png)

SVD使用奇异值分解来找出空间V，其中Σ也是一个对角矩阵，不过它对角线上的元素是奇异值，这也是SVD中用来衡量特征上的信息量的指标。

![Untitled](sklearn%E4%B8%AD%E7%9A%84%E9%99%8D%2081266/Untitled%201.png)

*class sklearn.decomposition.PCA (n_components=None, copy=True, whiten=False,svd_solver=’auto’, tol=0.0,iterated_power=’auto’, random_state=None)*

# 重要参数n_components：

n_components是我们降维后需要的维度，即降维后需要保留的特征数量，降维流程中第二步里需要确认的k值，一般输入[0, min(X.shape)]范围中的整数。

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
y = iris.target
X = iris.data
print(X.shape)

# 调用PCA
pca = PCA(n_components=2)  # 实例化
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取降维后的新矩阵
print(X_dr.shape)
# 也可以fit_transform一步到位
# X_dr = PCA(2).fit_transform(X)

# X_dr[y == 0, 0]  # 这里是布尔索引

colors = ['red', 'black', 'orange']
print(iris.target_names)
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0]
                , X_dr[y == i, 1]
                , alpha=.7  # 透明度
                , c=colors[i]
                , label=iris.target_names[i]
                )
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_)
# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
print(pca.explained_variance_ratio_)
# 大部分信息都被有效地集中在了第一个特征上
print(pca.explained_variance_ratio_.sum()) # 保留原数据的信息量占比
```

选择最好的n_components：累积可解释方差贡献率曲线

当参数n_components中不填写任何值，则默认返回min(X.shape)个特征，一般来说，样本量都会大于特征数目，所以什么都不填就相当于转换了新特征空间，但没有减少特征的个数。一般来说，不会使用这种输入方式。但我们却可以使用这种输入方式来画出累计可解释方差贡献率曲线，以此选择最好的n_components的整数取值。累积可解释方差贡献率曲线是一条以降维后保留的特征个数为横坐标，降维后新特征矩阵捕捉到的可解释方差贡献率为纵坐标的曲线，能够帮助我们决定n_components最好的取值。

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
y = iris.target
X = iris.data
print(X.shape)

# 调用PCA
pca = PCA(n_components=2)  # 实例化
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取降维后的新矩阵
print(X_dr.shape)
# 也可以fit_transform一步到位
# X_dr = PCA(2).fit_transform(X)

# X_dr[y == 0, 0]  # 这里是布尔索引

colors = ['red', 'black', 'orange']
print(iris.target_names)
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0]
                , X_dr[y == i, 1]
                , alpha=.7  # 透明度
                , c=colors[i]
                , label=iris.target_names[i]
                )
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_)
# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
print(pca.explained_variance_ratio_)
# 大部分信息都被有效地集中在了第一个特征上
print(pca.explained_variance_ratio_.sum())  # 保留原数据的信息量占比

import numpy as np

pca_line = PCA().fit(X)
print(pca_line.explained_variance_ratio_)
plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()
# 选择曲线突然变平缓的点为最佳维度
```

## 最大似然估计自选超参数

让PCA用最大似然估计(maximum likelihood estimation)自选超参数的方法，输入“mle”作为n_components的参数输入，就可以调用这种方法。

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
y = iris.target
X = iris.data
print(X.shape)

# 调用PCA

pca_mle = PCA(n_components="mle")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)
print(X_mle.shape)

# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca_mle.explained_variance_)
# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
print(pca_mle.explained_variance_ratio_)
# 大部分信息都被有效地集中在了第一个特征上
print(pca_mle.explained_variance_ratio_.sum())  # 保留原数据的信息量占比
```

## 按信息量占比选超参数

输入[0,1]之间的浮点数，并且让参数svd_solver =='full'，表示希望降维后的总解释性方差占比大于n_components指定的百分比，即是说，希望保留百分之多少的信息量。比如说，如果我们希望保留97%的信息量，就可以输入n_components = 0.97，PCA会自动选出能够让保留的信息量超过97%的特征数量。

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
y = iris.target
X = iris.data
print(X.shape)

# 调用PCA

pca_f = PCA(n_components=0.97,svd_solver="full")
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)
print(X_f.shape)

# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca_f.explained_variance_)
# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
print(pca_f.explained_variance_ratio_)
# 大部分信息都被有效地集中在了第一个特征上
print(pca_f.explained_variance_ratio_.sum())  # 保留原数据的信息量占比
```

svd_solver是奇异值分解器的意思,PCA和SVD涉及了大量的矩阵计算，两者都是运算量很大的模型，但其实，SVD有一种惊人的数学性质，即是它可以跳过数学神秘的宇宙，不计算协方差矩阵，直接找出一个新特征向量组成的n维空间。**奇异值分解可以不计算协方差矩阵等等结构复杂计算冗长的矩阵，就直接求出新特征空间和降维后的特征矩阵。**

简而言之，SVD在矩阵分解中的过程比PCA简单快速，虽然两个算法都走一样的分解流程，但SVD可以作弊耍赖直接算出V。但是遗憾的是，SVD的信息量衡量指标比较复杂，要理解”奇异值“远不如理解”方差“来得容易，因此，sklearn将降维流程拆成了两部分：一部分是计算特征空间V，由奇异值分解完成，另一部分是映射数据和求解新特征矩阵，由主成分分析完成，实现了用SVD的性质减少计算量，却让信息量的评估指标是方差。

# 重要参数svd_solver 与 random_state

参数svd_solver是在降维过程中，用来控制矩阵分解的一些细节的参数。
有四种模式可选："auto", "full", "arpack","randomized"，默认”auto"。

· "auto"：基于X.shape和n_components的默认策略来选择分解器：如果输入数据的尺寸大于500x500且要提取的特征数小于数据最小维度min(X.shape)的80％，就启用效率更高的”randomized“方法。

· "full"：从scipy.linalg.svd中调用标准的LAPACK分解器来生成精确完整的SVD，**适合数据量比较适中，计算时间充足的情况。**

· "arpack"：从scipy.sparse.linalg.svds调用ARPACK分解器来运行截断奇异值分解(SVD truncated)，分解时就将特征数量降到n_components中输入的数值k，**可以加快运算速度，适合特征矩阵很大的时候，但一般用于特征矩阵为稀疏矩阵的情况**，此过程包含一定的随机性。

· "randomized"，通过Halko等人的随机方法进行随机SVD。在"full"方法中，分解器会根据原始数据和输入的n_components值去计算和寻找符合需求的新特征向量，但是在"randomized"方法中，分解器会先生成多个随机向量，然后一一去检测这些随机向量中是否有任何一个符合我们的分解需求，如果符合，就保留这个随机向量，并基于这个随机向量来构建后续的向量空间。**适合特征矩阵巨大，计算量庞大的情况。**

通常我们就选用”auto“，不必对这个参数纠结太多。

重要属性components_

PCA与特征选择的区别，即特征选择后的特征矩阵是可解读的，而PCA降维后的特征矩阵式不可解
读的：PCA是将已存在的特征进行压缩，降维完毕后的特征不是原本的特征矩阵中的任何一个特征，而是通过某些方式组合起来的新特征。通常来说，**在新的特征矩阵生成之前，我们无法知晓PCA都建立了怎样的新特征向量，新特征矩阵生成之后也不具有可读性，**我们无法判断新特征矩阵的特征是从原数据中的什么特征组合而来，新特征虽然带有原始数据的信息，却已经不是原数据上代表着的含义了。

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

'''实例化数据集，探索数据'''
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.images.shape)
print(faces.data.shape)
X = faces.data

'''看看图像什么样？将原特征矩阵进行可视化'''
# 数据本身是图像，和数据本身只是数字，使用的可视化方法不同

# 创建画布和子图对象
fig, axes = plt.subplots(4, 5
                         , figsize=(8, 4)
                         , subplot_kw={"xticks": [], "yticks": []}  # 不要显示坐标轴
                         )
print(axes.flat)

enumerate(axes.flat)

# 填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i, :, :]
              , cmap="gray")  # 选择色彩的模式

'''建模降维，提取新特征空间矩阵'''
# 原本有2900维，我们现在来降到150维
pca = PCA(150).fit(X)
V = pca.components_
print(V.shape)

'''将新特征空间矩阵可视化'''
fig, axes = plt.subplots(3, 8, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(62, 47), cmap="gray")

plt.show()
```

## 重要接口inverse_transform

降维的目的之一就是希望抛弃掉对模型带来负面影响的特征，而我们相信，带有效信息的特征的方差应该是远大于噪音的，所以相比噪音，有效的特征所带的信息应该不会在PCA过程中被大量抛弃。inverse_transform能够在不恢复原始数据的情况下，将降维后的数据返回到原本的高维空间，即是说能够实现”保证维度，但去掉方差很小特征所带的信息“。利用inverse_transform的这个性质，我们能够实现噪音过滤。

迷你案例：用PCA做噪音过滤

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
print(digits.data.shape)

'''定义画图函数'''
def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4)
                             , subplot_kw={"xticks": [], "yticks": []}
                             )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap="binary")

plot_digits(digits.data)

'''为数据加上噪音'''
np.random.RandomState(42)
# 在指定的数据集中，随机抽取服从正态分布的数据
# 两个参数，分别是指定的数据集，和抽取出来的正太分布的方差
noisy = np.random.normal(digits.data, 2)
plot_digits(noisy)

'''降维'''
pca = PCA(0.5).fit(noisy)
X_dr = pca.transform(noisy)
print(X_dr.shape)

'''逆转降维结果，实现降噪'''
without_noise = pca.inverse_transform(X_dr)
plot_digits(without_noise)
plt.show()
```

[案例：PCA对手写数字数据集的降维](sklearn%E4%B8%AD%E7%9A%84%E9%99%8D%2081266/%E6%A1%88%E4%BE%8B%EF%BC%9APCA%E5%AF%B9%E6%89%8B%E5%86%99%E6%95%B0%203b8a6.md)