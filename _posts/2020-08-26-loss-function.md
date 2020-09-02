---
layout: post
title:  "Loss Function: Why |y_true-y_predict| is not enough? "
date:   2020-04-24 22:21:49 +1000
categories: ML
permalink: /:title.html
---
Loss functions guide the direction of the training process via derivatives w.r.t parameters $W$ (normally I ignore the bias term $b$ since $W$ do the job of transforming input or we can add an additional input of 1 and $w_{n+1}$ could be integrated with $W$).  Intuitively, loss function will tell us how  wrong the current model predicts in contrast to the true values of samples.  However,  if this is the case, we just need to find parameter $W$  satisfying .$\sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2} = 0$. Normally, we average the square error: $ \mathrm{MSE}=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}$.

There is a trick here: sample data won' t span the entire space so that a set of parameters could satisfy the sample points, e.g. classifying all sample points correctly. By the way, this post would focus on classification instead of regression.    Another thing is the bias of sample





# Binary Classification

For example, for passengers in Titanic, they are either survived or not survived which can be represented as **a variable** with values {0, 1}. Specifically,  {0: Not Survived, 1 : Survived}. 

* Terminology: **dummy variable** 

    I think the reason it is call dummy is because compared to original meanings of binary outcomes(survived or not survived), it seems to be dummy.

## It can be transferred into a probability problem. -> Get probability distribution to make decisions

* one predicted value between 0 and 1 could be interpreted as probability to determine the result by setting a threshold, normally 0.5. Statistically, I prefer to interpret it as two probabilities p and q for each class(Probability Distribution). But instead of separately estimating q and normalize them. It defines q = 1 - p. Then this threshold 0.5 can be considered as selecting the most probable class.
    * the predicted probability is returned by a hypothesis function to model the probability that y = 1, given x, parameterized by θ, written as: 
       $$ h(x) = p=Pr(y = 1|x; θ) $$
    * The most popular hypothesis function is logistic regression. But we also could use linear probability model.
        + Hypothesis Function 1: linear model or in this scenario call [Linear Probability Model](https://en.wikipedia.org/wiki/Dummy_variable_(statistics)#Linear_probability_model). BUT  probabilities are not linear in terms of $X_i$
        + Hypothesis Function 2: Logistic Regression. 
        
        Consider the outcome of linear function as **log of odd**/[logit](https://en.wikipedia.org/wiki/Logit) of categorical outcomes $\ln \frac{p}{1-p}$ with assumption of the log odds of p is linearly related to the independent variable X. So, it combines **Linear Function**(right) and **Logit function/log-odd**(left) 
        
        [\ \ln \frac{p}{1-p} = \theta^{T} X \] 
        
        Inversed the logit function, it become **logistic/sigmoid function** for the probability which is considered as hypothesis
        $$  h(x)=p=\frac{1}{1+e^{-\theta^{T} X}} $$
         Decision boundary can be described as: Predict 1, if θᵀx ≥ 0 → h(x) ≥ 0.5; Predict 0, if θᵀx < 0 → h(x) < 0.5.
    
## Lastly, for getting optimized parameters θ of the model, we need define the loss function to do backward pass.



## Generalize to Multi-class Classification: Same as Binary Case, classification is made by estimating parameters of probability distribution $ p(y | X, \theta ) $.

*  Since now there are more than 2 possible outcomes for $y$. Instead of Logistic/sigmoid Function, here Softmax Function is applied to get probability distribution of each class, i.e. probabilities of all the classes would sum up to 1, i.e. normalize scores of each class $s(x)$. 

* Normally, the score $s(x)$ is also a linear function. Then the whole algorithm is called Softmax Regression or [Multinomial Logistic Regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

* Also same as before, use `cross entropy loss function`.



## How to define Loss Function?

* In statistics, we can model its probability distribution as **Bernoulli distribution**. The PDF is as followed:  It can be intuitively understood if y=1, it gives probability of Survived, I.E. p; if y=0, it gives probability of not Survived, I.E. (1-p).

    $$f(y ; p)=p^{y}(1-p)^{1-y} \quad \text { for } y \in\{0,1\}$$
    
    Related to the case of logistic regression, it would be: $$ Pr\left(y^{(i)} | x^{(i)} , \theta\right)=p^{y^{(i)}} (1-p)^{1-y^{(i)}}$$ where $$p = h_\theta=\frac{1}{1+e^{-\theta^{T} X}} $$
    
    To estimate `p` which also relates to a set of $\theta$, we can use `Maximumn Likelihood Estimator` or `MSE`.
    
* How `MSE` work? 
    + Simply speaking, it gets the most possible `p` or $\theta$ given training sample -> the probability of training sample given model/distribution. i.e. maximize $  L(\theta)=Pr(y | X , \theta)$. By assuming Independent and Identical Distribution(IID), it would be:
$ L(\theta)=\prod_{i=1}^{m} Pr\left(y^{(i)} | x^{(i)} , \theta\right) $
    + `MSE` could estimate parameters for different probability distribution. For binary classification, it would be: 
    $$L(\theta) = \prod_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)\right)^{y^{(i)}}\left(1-h_{\theta}\left(x^{(i)}\right)\right)^{1-y^{(i)}}$$
    + Normally, we simplify it by using `Log Likelihood` which maintains the same optimal position but turns product operation into summation.
    $$\begin{aligned} -\ell(\theta) &=-\log L(\theta) \\ &=-\sum_{i=1}^{m} y^{(i)} \log h\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h\left(x^{(i)}\right)\right) \end{aligned}$$
    + PS1: Likelihood vs Probability: In statistics, $ p(x<0.5|\mu=0, \sigma^2=1) $ is the area under a fixed PDF/probability model. $L(\theta|x_i)$ is the y-axis value/probability on PDF for fixed data points with can-be-moved distribution.
    + PS2: if we use it estimate parameters of linear regression, the optimal position is the same as the result of Mean Square Error. So MLE provides a theoretical support for machine learning algorithms.
    + PS3: However, `Maximum Likelihhood` is not the only type of loss function. Others include `MAP(Maximum A Posteriori)`. From [Wikipedia](https://en.wikipedia.org/wiki/Multinomial_logistic_regression), "it is an extension of maximum likelihood using regularization of the weights to prevent pathological solutions (usually a squared regularizing function, which is equivalent to placing a zero-mean Gaussian prior distribution on the weights, but other distributions are also possible)". More see this [post](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83)
   
* Finish the Story: Then we can use Stochastic Gradient Descent to optimise this equation to estimate a set of $\theta$.
        
  
    

     
  


​     