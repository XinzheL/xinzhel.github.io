* Understand the choice of loss function
    * 学习过程
        1. 信息流forward propagation，直到输出端；

        2. 定义损失函数L(x, y | theta)；

        3. 误差信号back propagation。采用数学理论中的“链式法则”，求L(x, y | theta)关于参数theta的梯度；

        4. 利用最优化方法（比如随机梯度下降法），进行参数更新；

        重复步骤3、4，直到收敛为止；
     * 损失函数: 常见的有均方误差(error of mean square)、最大似然误差(maximum likelihood estimate)、最大后验概率(maximum posterior probability)、交叉熵损失函数(cross entropy loss)。一般地，一个机器学习模型选择哪种损失函数，是凭借经验而定的，没有什么特定的标准。具体来说，

        1. **均方误差**是一种较早的损失函数定义方法，它衡量的是两个分布对应维度的差异性之和。说点题外话，与之非常接近的一种相似性度量标准“**余弦角**”，则衡量的是两个分布整体的相似性，也即把两个向量分别作为一个整体，计算出的夹角作为其相似性大小的判断依据;
2. **最大似然误差**是从概率的角度，求解出能完美拟合训练样例的模型参数theta，使得概率p(y | x, theta)最大化；
        3. **最大化后验概率**Maximum A Posterior，即使得概率p(theta | x, y)最大化，实际上也等价于带正则化项的最大似然概率（详细的数学推导可以参见Bishop 的Pattern Recognition And Machine Learning），它考虑了先验信息，通过对参数值的大小进行约束来防止“过拟合”；
4. 交叉熵损失函数，衡量的是两个分布p、q的相似性。在给定集合上两个分布p和q的cross entropy定义如下：
         $$H(p, q)=\mathbf{E}_{p}[-\log q]=H(p)+D_{\mathrm{KL}}(p \| q)$$
 在机器学习应用中，p一般表示样例的标签的真实分布，为确定值，故最小化交叉熵和最小化KL-devergence是等价的，只不过之间相差了一个常数  
        KL Divergence: $D_{\mathrm{KL}}(Y \| \hat{Y})=-\sum_{i} y^{(\mathrm{i})} \log \frac{\hat{y}^{(\mathrm{i})}}{y^{(\mathrm{i})}}$
    
* Least Square vs MSE