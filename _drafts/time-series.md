---
layout: "post"
title: "Time Series"
---



# Basis of Time Series

* 随机时间序列($x_1, x_2,...,x_t$) vs 观察值序列

  * 一般我们所能得到的都是观察值序列，即随机时间序列的一次采样;
  * 一个随机变量的统计特征可以由它的分布函数或密度函数来决定；同样，随机变量族$X_t$ 的所有统计特征完全可以由他们的联合分布函数或者[联合密度函数](https://en.wikipedia.org/wiki/Stationary_process#Joint_strict-sense_stationarity)来决定;
  * 通过样本的性质来推断总体的参数或分布，包括：均值，方差，总体分布，边缘分布，联合分布等。
  
* 统计时间序列分析方法
  * 频谱分析方法：假设任何一种无趋势的时间序列都可以被分解为若干不同频率的周期波动（发展：傅里叶分析/傅里叶变换/最大熵谱估计）
  * 时域分析方法: 从序列自相关的角度揭示时间序列的发展规律（惯性）。
  * 步骤：1. 考察观察值序列的特征；2.选择合适的拟合model；3.确定model口径；4.检验优化model；5.用拟合好的model来推断序列其它的统计性质或预测序列将来的发展。
  * [时频分析]([https://en.wikipedia.org/wiki/Time%E2%80%93frequency_analysis](https://en.wikipedia.org/wiki/Time–frequency_analysis))
  
* Types of Questions: 
  * Given the spectra from different birds
    * Classification: Which bird is this?
    * Prediction: How long will the song continue?
    * Outlier Detection: Is this bird sick?
    * Segmentation: What phases does this song have?
  * Given the signal of stock price
    * Prediction: Will the stock go up or down?
    * Classification: What type of stock is this(e.g. risky)?
    * Outlier Detection: Is the behavior abnormal?
  
* 白噪声：对所有时间其相关系数为0的随机过程

  $\forall y_t r(y_t, y_{t-k})=0$ 

## 时间序列的预处理
平稳性检验和随机性检验
### 平稳性
1. 为什么要求数据的平稳性？
  * 类比机器学习中的i.i.d.：独立同分布是利用大数定律的前提。例如使用样本($x_1, x_2, ..., x_n$)估计随机变量$X \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$的均值：$(x_1+x_2+...+x_n)/n$ 
    * $\frac{1}{N} \sum_{i=1}^{N} L\left(y_{i}, f\left(x_{i}\right)\right)$ -> $\int_{X, Y} L(y, f(x)) P(x, y) d x d y$
  * 时间序列无法这么做，因为时间序列只有一条轨道($y_1, y_2, ..., y_t$) 。但是如果满足时间遍历性，就可以类似的通过i.i.d.来都总体参数进行估计。
  * 时间遍历性：平稳性

2. 如何变成平稳的？做差分。
```
np.diff(ts)
```
3. 平稳性检验单位根检验(0.05~0.01)
```
from statsmodels.stats.diagnostic import unitroot_adf
unitroot_adf(ts)[1] # p value
```



# 时间序列预测
* 目标：点预测和区间预测
  
  * e.g. 使用均值做点预测，因为 正态分布中均值概率最大
  
* 什么样的时间序列时可预测的？
  * 影响因素已知
  * 大量可用的数据
  * 预测不会反向影响我们试图预测的事物（内生性问题）
    * VAR模型没有内生性问题
  
* 什么样的时间序列是可定量建模的？
  * 关于过去的数据是可用的；
  * 有理由假设过去的一些模式会在未来延续下去。比如通过过去80个月的数据预测CoVid对经济的影响是不可行的由于过去80个月并没有类似的模式。或者可以考虑使用百年前西班牙大流感的数据，当然大环境(人口，医疗水平等)的改变会影响数据的可用性。
  
* 模型选择
    * 简单规则模型(W：窗口)

      * 均值法: $\hat{y}_{T+h}=\frac{1}{W} \sum_{i=1}^{W} y_{T-W+i} $
      * 朴素法:  $\hat{y}_{T+h} = y_T$
      * 季节性朴素预测法： $\hat{y}_{T+h}=y_{T+h-k m}$ （机器学习中可以把这个规则预测作为特征）
      * 漂移法：$y_{T+h}=y_{T}+h \frac{y_{T}-y_{T-W}}{W}$ 时间长度*增长率

    * 指数平均模型 - 一阶指数平滑

      * $\hat{y_{t+1}}=\alpha y_{t}+\alpha(1-\alpha) y_{t-1}+\alpha(1-\alpha)^{2} y_{t-2}+\cdots$

      * 递归表达：$\hat{y}_{i+1}=\alpha y_{t}+(1-\alpha) \hat{y_{t}}$

      * 分量表达

        * Forecast equation: $\hat{y}_{t+h|t}=\ell_t$
        * Smoothing equation: $\ell_{t}=\alpha y_{t}+(1-\alpha) \ell_{t-1}$

      * 通过分量表达推广到二阶指数平滑，三阶指数平滑...

        ```python
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        alpha = 0.15
        simpleExpSmooth_model = SimpleExpSmoothing(ts).fit(smoothing_level=alpha, optimized=False)
        simpleExpSmooth_model.forecast(28)
        ```

    * 指数平均模型 - 二阶指数平滑（趋势）

      * Forecast equation: 

        * 不带阻尼：
          * $\hat{y}_{t+h|t}=\ell_t+hb_t$
        * Level equation $\ell_{t}=\alpha y_{t}+(1-\alpha)\left(\ell_{t-1}+b_{t-1}\right)$$
          * Trend equation$b_{t}=\beta^{*}\left(\ell_{t}-\ell_{t-1}\right)+\left(1-\beta^{*}\right) b_{t-1}$
        * 带阻尼：
          *  $\hat{y}_{t+h t}=\ell_{t}+\left(\phi+\phi^{2}+\cdots+\phi^{h}\right) b_{t}$
          * Level equation $\ell_{t}=\alpha y_{t}+(1-\alpha)\left(\ell_{t-1}+\phi b_{t-1}\right)$$
          * Trend equation$b_{t}=\beta^{*}\left(\ell_{t}-\ell_{t-1}\right)+\left(1-\beta^{*}\right) \phi b_{t-1}$
        
        ```python
        from statsmodels.tsa.holtwinters import Holt
        DaMExpSmooth_model = Holt(ts, damped=True).fit(smoothing_level=0.9, smoothing_slope=0.8)
        ```

    * 指数平均模型 - 三阶指数平滑（趋势+季节)

      * $\hat{y}_{t+h | t}=\ell_{t}+h b_{t}+s_{t-m+h_{m}^{+}}$
      * $\ell_{t}=\alpha\left(y_{t}-s_{t-m}\right)+(1-\alpha)\left(\ell_{t-1}+b_{t-1}\right)$
      * $\begin{array}{l}
        b_{t}=\beta^{*}\left(\ell_{t}-\ell_{t-1}\right)+\left(1-\beta^{*}\right) b_{t-1} \\
        s_{t}=\gamma\left(y_{t}-\ell_{t-1}-b_{t-1}\right)+(1-\gamma) s_{t-m}
        \end{array}$

      ```python
      from statsmodels.tsa.holtwinters import ExponentialSmoothing
      HW_ExpSmooth_model = ExponentialSmoothing(ts, seasonal_periods=7, trend='add', seasonal='add', damped=True).fit()
      ```

    * 指数平均拓展 - sgd with momentum

    * 自回归模型

      * 回归模型、自回归模型和动态回归模型的区别

      * 和指数平滑模型比较：

        * 指数平滑模型针对于数据中的趋势和季节性
        * 自回归模型描述数据的自回归性

      * 通过统计的方法对参数进行估计，对时间序列要求平稳性

        

    * AR

      * $y_{t}=c+\phi_{1} y_{t-1}+\phi_{2} y_{t-2}+\cdots \phi_{p} y_{t-p}+\epsilon_{t}$
      * $\phi$通过统计学习估计出来的，是有约束的，随着p的不同，$\phi$会变得很复杂

    * MA

      * $y_{t}=c+\epsilon_{t}+\theta_{1} \epsilon_{t-1}+\theta_{2} \epsilon_{t-2}+\theta_{3} \epsilon_{t-3} \cdots \theta_{q} \epsilon_{t-q}$

    * ARIMA(p,d,q)

      * $y_{t}=c+\phi_{1} y_{t-1}+\phi_{2} y_{t-2}+\cdots \phi_{p} y_{t-p}+\epsilon_{t}+\theta_{1} \epsilon_{t-1}+\theta_{2} \epsilon_{t-2}+\theta_{3} \epsilon_{t-3} \cdots \theta_{q} \epsilon_{t-q}$

      * d: 表示做多少次差分保证平稳性

      * AR 模型和机器学习模型里面的LR再参数的学习上是不是一样的？拟牛顿法，梯度下降都可以

      * 步骤：1.序列平稳； 2. 定阶数；3.求参数

      * ![image-20200511152359918](C:\Users\sergi\AppData\Roaming\Typora\typora-user-images\image-20200511152359918.png)

        ```python
        from pmdarima import auto_arima # 自动的定阶数
        arima_model = auto_arima(ts, start_p=0, start_q=0, max_p=10, max_q=5, seasonal=False, d=None, trace=True, random_state=666, error_action='ignore'，suppress_warnings=True, stepwise=True)
        # SARIMA->seasonal=True 
        # 非打比赛角度，考虑模型参数的复杂度：aic， bic
        arima_model.predict(n_periods=28)
        ```

    * SARIMA( consider seansonal)

    * 机器学习

      * 前面讲的算法都没考虑外生变量（SARIMAX会考虑，类似于动态回归，但是一般也比ML好）

* [时间序列分解](https://otexts.com/fpp3/decomposition.html)

    * SLT
        * 分类:  $\text {Score}_{T}=\max \left(0, 1-\frac{\operatorname{Var}\left(R_{t}\right)}{\operatorname{Var}\left(T_{t}+R_{t}\right)}\right)$ 和$\text {Score}_{s}=\max \left(0, 1-\frac{\operatorname{Var}\left(R_{t}\right)}{\operatorname{Var}\left(S_{t}+R_{t}\right)}\right)$

    ```
    import statsmodels.api as sm
    rd = sm.seasonal_decompose(ts, freq=7) # 'freq': check ACF
    ```

    

    * 移动平均 -> 加权移动平均 -> 自己学权重（一维卷积/CNN）













