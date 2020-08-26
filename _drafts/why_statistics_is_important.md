---
layout: "post"
title: "Why distribution is important for modelling?"
---

"Each theoretical domain provides a ready-to-use package for solving problem". --- Xinzhe Li

When models learn knowledge from data to represent the population, it has to e



How different kinds of distributions are useful for building reasonable ML models?

* Distributions represent data-generating mechanism:
  * Exampe: Thinking about predicting whether faults happen to an engine for one observation, noted as `Occur` .  
  * Explain: We know faults rarely happen to an engine that can be represented by probability. Let's say `Pr(Occur=yes)=0.01` . Therefore, the distribution of `Occur` can induce such knowledge into our model. It normally are represented by probabilities for different values of this variable, i.e. `Pr(Occur=yes)` and `Pr(Occur=no)` (Terminology: Probability Mass Function & Probability Density Function). Next, take one of ML algorithms as example to see how it affect modelling results.
  * Logistic Regression: Normally, ML algorithms predict the vaule of `Occur`. Interestingly, logistic regression models `Pr(Occur=yes)` as responding variable which is the probability of a yes-no question. [Bernoulli Distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) is theorised for describing such probability variable with parameter `p`. The problem can be transformed as a problem for estimating `p` using MLE(see [this post](logistic_regression) for more).
* When this kind of distributionThe importance of i.i.d
  * Example: Thinking about predicting whether faults happen to an engine for one observation at given points(Time series)
  * 

i.i.d: 

* Independent: observations are unrelated.
* indentically distributed: Mechanism to generate data is the same. That means that probabilities associated with the distribution are the same.

As Law of Large Number, Central Limit Theorem



Since observations in a time series are not independent 