# Arsenal

整理了在分析和建模过程中常用的工具


## 相关性分析

### Pearson相关系数

主要检验两个变量之间是否存在线性相关性

$$
\rho_{X, Y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{X} \sigma_{Y}}=\frac{E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}
$$

### MIC

>Detecting Novel Associations in Large Data Sets

mic 的想法是针对两个变量之间的关系离散在二维空间中，并且使用散点图来表示，将当前二维空间在 x,y 方向分别划分为一定的区间数，然后查看当前的散点在各个方格中落入的情况，这就是联合概率的计算，这样就解决了在互信息中的联合概率难求的问题。下面的公式给出 mic 的计算公式：

$$
\operatorname{mic}(x ; y)=\max _{a * b<B} \frac{I(x ; y)}{\log _{2} \min (a, b)}
$$

$$
I(x ; y)=\int p(x, y) \log _{2} \frac{p(x, y)}{p(x) p(y)} \mathrm{d} x \mathrm{d} y
$$

MIC 可以计算非线性相关的变量之间的相关度

实现[Python]: [minepy](https://minepy.readthedocs.io/en/latest/)

## 异常值分析

### 1 维数据

- 3 sigma

### 2 维数据

### 多维数据

## 特征筛选

### L1 正则

正则化是机器学习中最常用的正则化方法，通过约束参数的 $l_1$ 和 $l_2$ 范数来减小模型在训练数据集上的过拟合现象。

$$
\theta^{*}=\underset{\theta}{\arg \min } \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}\left(y^{(n)}, f\left(\mathbf{x}^{(n)}, \theta\right)\right)+\lambda \ell_{p}(\theta)
$$

$l_1$ 范数的约束通常会使得最优解位于坐标轴上，而从使得最终的参数为稀疏性向量。

### sequential feature 算法

该算法分为向前和向后两种，核心是使用贪心算法：

- 向前：从空集开始，每次选择能使得评价函数 $J(X)$ 最优的一个特征 $X$ 加入
- 向后：从特征全集开始，每次选择使评价函数 $J(X)$ 最优的特征 $X$ 剔除

该方法不一定会得到最优的特征组合，但整体时间复杂度较低。

实现[Python]: [mlxtend.feature_selection.SequentialFeatureSelector](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#api)

### Recursive feature elimination (RFE)

递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一边，在剩余的特征上重复这个过程，直到遍历所有特征。这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。

实现[Python]: [sklearn.feature_selection.RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE)  
 [sklearn.feature_selection.RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)

## 自动调参

### 网格搜索

网格搜索(grid search)是一种通过尝试所有超参数的组合来寻址合适一 组超参数配置的方法。

一般而言，对于连续的超参数，我们不能按等间隔的方式进行离散化，需要根据超参数自身的特点进行离散化。

实现[Python]: [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)

### 贝叶斯优化

>Algorithms for hyper-parameter optimization

>Sequential model-based op- timization for general algorithm configuration

贝叶斯优化是一种自适应的超参数搜索方法，根据当前已经试验的超参数组合，来预 测下一个可能带来最大收益的组合。一种比较比较常用的贝叶斯优化方法为时序模型优化(SMBO)。

假设超参数优化的函数 $f (x)$ 服从高斯过程，则 $p(f (x)|x)$ 为一个正态分 布。贝叶斯优化过程是根据已有的N组试验结果 $\mathcal{H}=\left\{\mathbf{x}_{n}, y_{n}\right\}_{n=1}^{N}$ ( $y_n$ 为 $f(x_n)$ 的观测值)来建模高斯过程，并计算 $f(x)$ 的后验分布$p_{\mathcal{G} \mathcal{P}}(f(\mathbf{x}) | \mathbf{x}, \mathcal{H})$。

贝叶斯优化的一个缺点是高斯过程建模需要计算协方差矩阵的逆，时间复杂度是 $O(n^3 )$，**因此不能很好地处理高维情况。**

实现[Python]: [hyperopt](https://github.com/hyperopt/hyperopt)

[文档地址](https://github.com/hyperopt/hyperopt/wiki)

## 时序预测

### ARIMA

### STL 分解

### Fb Prophet

[官网地址](https://facebook.github.io/prophet/)，Facebook 开源的时序预测工具

## 时序滤波

## 稳定性检测

### ADF test

在统计学中 augmented Dickey–Fuller test 用于检测时序数据的平稳性和周期性检测方法，其原假设为时序存在单位根，并计算满足该假设的 p 值， 如果 p 值过小，则否定原假设——时序数据不存在周期性。具体方法如下：


创建回归模型，对时序数据进行拟合，模型的具体方程为：
$$
\Delta y_{t}=\alpha+\beta t+\gamma y_{t-1}+\delta_{1} \Delta y_{t-1}+\cdots+\delta_{p-1} \Delta y_{t-p+1}+\varepsilon_{t}
$$


进行单位根检验，计算DF值：
$$
D F_{\tau}=\frac{\hat{\gamma}}{S E(\hat{\gamma})}
$$
并将DF值与 Dickey–Fuller Test 的值相比较，如果 DF 值小于比较值，则否定原假设，数据不存在周期性。

**ps**： ADF test 不支持含有趋势性的时序数据检验，因此在检测前必须去除趋势性

实现[Python]: 

## 时序异常

### EDM 算法

>Leveraging Cloud Data to Mitigate User Experience from ‘Breaking Bad’

实现[R]: [BreakoutDetection](https://github.com/twitter/BreakoutDetection)


## 时序变点检测

### PELT 算法

>Optimal detection of changepoints with a linear computational cost

### Bayes
>Modeling Changing Dependency Structure in Multivariate Time Series

时间复杂度较高，如果对性能有需求的场景，不建议使用

实现[Python]: [bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)

## 分布拟合

### KS 检验

Kolmogorov-Smirnov是比较一个频率分布 $f(x)$ 与理论分布 $g(x)$ 或者两个观测值分布的检验方法。其原假设 $H_0:$ 两个数据分布一致或者数据符合理论分布。$D=max| f(x)- g(x) |$，当实际观测值 $D>D(n,\alpha)$ 则拒绝 $H_0$，否则则接受 $H_0$ 假设。

实现[Python]：[scipy.stats.ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)

## 部署

### Docker

### Flask

## 性能提升

### pypy

### numba