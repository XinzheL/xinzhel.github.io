---
layout: post
title:  "NN = Lego pieces with transformation power🧱"
date:   2020-04-24 22:21:49 +1000
categories: ML
permalink: /:title.html
---

Note: In case that you would like to learn by practice, you could run the code by yourself online by forking [my corresponding notebook](https://www.kaggle.com/sergioli212/implement-simple-nn-from-scratch) on Kaggle.



You want to build Chopper with Lego pieces with transformation power which means if a triangle piece does not satisfy you for the cute deer horn, it can be transform to the designed shape you need. However, we know only triangle pieces (like linear transformation) cannot suit the need for building complex "things" so that we need some irregularly shaped pieces (like non-linear transformation with activation function). This is exactly the case how we resolve problems using neural network. The transformation power comes from transformation matrix $W$. However, just like you can hardly predict man's heigh by his age and richness, the transformation power is not a limitless magic but depends which Lego pieces are provided.

![image-20200828150224459](https://i.loli.net/2020/08/28/Qm7YkBI8OKj5LwH.png)

Notice that the story here is a complete neural network architecture (**I.E. aiming to introduce Lego pieces**). However, nothing relates to models which could be tailored for particular problem without introducing loss function  and particular sample data(**I.E. what you want to build, maybe Chopper😍**) , let alone intelligence without further introducing automatic training algorithms(**I.E. building a  Doraemon 😘**).  The reason I would like to underline this point is because the novice tend to mix up all concepts or algorithms as parts of neural network or deep learning. It would hurt for flexibly implementing them. 



Anyway, it is just a story of neural network architecture.



# First Piece of Lego: Feed-forward NN

The real power of deep neural networks come from non-linear transformation with **activation function** and multiple layers and units which extracted abundant and may-be-useful information. However, linear transformation provides some properties which give convenience for intuitively understanding the nature of neural networks. So begin with linear transformation now. 

The linear transformation could be represented by the following linear function mapping input X to the output Y by finding the parameters W, determined by loss function: 

Let  $m$ = the number of examples, $n_i$ = the input dimension, $n_o$ = the output dimension, then 

$\hat{Y} = W \cdot X$  where $\mathbf{X} \in \mathbb{R}^{n_i \times m}$ and $\mathbf{W} \in \mathbb{R}^{n_o \times n_i}$ and $\mathbf{\hat{Y}} \in \mathbb{R}^{n_o \times m}$ 

which is equivalent to 

$\hat{Y} = X \cdot W$  where $\mathbf{X} \in \mathbb{R}^{m \times n_i}$ and $\mathbf{W} \in \mathbb{R}^{n_i \times n_o}$ and $\mathbf{\hat{Y}} \in \mathbb{R}^{m \times n_o}$ (This form conforms to the implementation in Tensorflow).

We use the first notation for the following description since it makes more sense for me to imagine it as the transformation in the high-dimension space. Normally, a bias term $b\in \mathbb{R}^{n_o \times 1}$ would be added so that $\hat{Y} = W \cdot X + b$.

The following example one layer Neural Network is indeed a logistic regression where I construct two units for estimating the probabilities of each class instead of deriving the probability of another class via [Complement Rule](https://brilliant.org/wiki/probability-by-complement/#:~:text=in%20the%20experiment.-,Complement%20Rule,or%20the%20other%20must%20occur).



```python
import numpy as np

def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)

m = 5
n_i = 10
n_o = 2


np.random.seed(1)

# generate experimental data
X = np.random.randn(n_i,m)

# initialize parameters: W, b
W = np.random.randn(n_o,n_i)
b = np.random.randn(n_o,1)
    
# transformation
Z = np.dot(W, X)
Y_hat = softmax(Z, axis=0)
```



Actually, above is the whole process of forward pass of neural network. The essence of neural network is $W$ because the only output of training process is the suitable $W$.  The rest of work is to determine $W$ by resolving optimization problems of well-defined [**loss function**](2020-08-26-loss-function.md) according to some assumptions which belongs to the backward pass.

Let alone loss function and backward pass of neural network, there are something missing for constructing the complete model so far.  In another word, is it possible to just rebuild the model with just trained parameters $W$ and $b$?  It lacks **model architecture**, including activation function in each layer, hyperparameter like learning rate, or special layer information like normalization, pooling, recurrent layer, convolution layer and so on.



# Other Lego Pieces with various shapes

Luckily, our Beautiful World is built on space and time which introduce awesome properties into data(e.g. Pictures, Text, Audio,Vedio and sensor readings and so on). The most important property is that positions, I.E. a sequence in 1-D space or an image in 2-D space. Specifically, the trend of temperature can only be reflected along sequential steps in order or images of human face could only be recognized when the nose is put between eyes and the mouth(otherwise, it would be recognized as the monster, actually nothing.) Even though these positional relations may be extracted by human as features and then previous standard Lego pieces may work, it just may not be well-thought-out and not intellgent when it comes to a bulk of data. Therefore, some NN architecture could be constructed to consider the nature of positional relations, e.g. Convolution NN and Recurrent NN. In essence, they are just another type of transformation or pieces of Lego for constructing more beautiful toys.

## Recurrent NN

Recurrent Layer is tailored to sequential data, i.e. the input would $x = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle})$ over $T_x$ time steps. 

![image-20200828151632260](https://i.loli.net/2020/08/28/jxZdgy9wXozc8Ff.png)

Depending on types of tasks, the ouput could be the sequence $y = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_y \rangle})$ or not. The following example assumes $T_x = T_y$


Let $a^{<i>}$ as the output of the i-th RNN cell and it would integrate information in the previous i time steps. The last output $a^{<T_x>}$ should contains the whole sequential information and could be the final output of the RNN layer. 

There are 3 shared transformation matrix happening to one recurrent layer: 
* $W_{aa}$: Control the transformation at time dimension.
* $W_{ax}$: Transform the feature dimension just like normal Dense Layer
* $W_{ya}$: Transform the  feature dimension and time dimension for targeting output. As discussed before, this would be determined by the type of tasks.



```python
np.random.seed(1)

X = np.random.randn(3,10,4)


n_x, m, T_x = X.shape # 3, 10, 4

n_y, n_a = 2, 5

# initialize parameters: Waa,  Wax, Wya,  ba, by}
a0 = np.random.randn(n_a,m)
Waa = np.random.randn(n_a,n_a)
Wax = np.random.randn(n_a,n_x)
Wya = np.random.randn(n_y, n_a)
ba = np.random.randn(n_a,1)
by = np.random.randn(n_y,1)

# initialize outputs: "a" and "y" 
a = np.zeros((n_a, m, T_x))
Y_pred = np.zeros((n_y, m, T_x))

# loop over all time-steps for each cell
a_prev = a0
for t in range(T_x):
    # compute next activation state/hidden state: W_aa*a_t-1
    a[:,:,t] = np.tanh(Wax.dot(X[:,:,t]) + Waa.dot(a_prev) + ba)
    # compute output of the current cell using the formula given above
    Y_pred[:,:,t] = softmax(np.dot(Wya, a[:,:,t]) + by)
    
    # save for (t+1) round
    a_prev = a[:,:,t]
    
    

print("a.shape = ", a.shape)
print("y_pred.shape = ", Y_pred.shape)
    

```

Normally, the last output $a^{<T_x>}$ could be used  the whole sequential information SO THAT the time dimension is eliminated and sequential information is extracted into  $a^{<T_x>}$. Note that the shape does not include time dimension as followed.

```python
print("Input shape ", X.shape)
print("After RNN layer, output shape ", a[:, :, 3].shape)
```


## LSTM

However, since transformation matrix $W_{aa}$ is calculated by gradient descend algorithm, [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) would occur which cause the information of far previous steps cannot flow into $a^<t>$. Therefore, another information pipe$C^{t}$ for memory state. The point is that there are no transformation $W$ directly working on $C^{<t>}$, i.e. Dot product for transformation only works for $a^{<t>}$, same as normal RNN. So, if so, how past information could flow into $C^{<t>}$? 

1 Gate to determine forgotten information from previous memory state $C^{<t-1>}$ and 1 Gate to determine gained information, then add. (Add operation will not cause multiplicative derivative if you know calculus property.)

Also, for maintaining output $a^{<t>}$ for transformation in the next step, Gate 3 would be used to filter information from $C^{<t>}$. The compact function is shown as below.

$$
\left(\begin{array}{c}
\mathbf{i}^{<t>} \\
\mathbf{f}^{<t>} \\
\mathbf{o}^{<t>} \\
\tilde{C}^{\prime}
\end{array}\right)=\left(\begin{array}{c}
\sigma \\
\sigma \\
\sigma \\
\tanh
\end{array}\right) \mathbf{W}\left(\begin{array}{c}
\mathbf{X}^{<t>} \\
\mathbf{a}^{<t-1>}
\end{array}\right)$$

The details are summarised from famous [Chrostopher Olah post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). Note that the syntax is different from what I use. But as long as you understand the point, it would be easy to understand in any way.
<img src="https://i.loli.net/2020/08/28/sqwj2Wkn8eYfHgX.png" width=650>





```python
np.random.seed(1)

X = np.random.randn(3,10,4)


n_x, m, T_x = X.shape # 3, 10, 4
n_y, n_a = 2, 5

# initialize parameters
np.random.seed(1)
Wf = np.random.randn(n_a, n_a+n_x)
bf = np.random.randn(n_a,1)
Wi = np.random.randn(n_a, n_a+n_x)
bi = np.random.randn(n_a,1)
Wo = np.random.randn(n_a, n_a+n_x)
bo = np.random.randn(n_a,1)
Wc = np.random.randn(n_a, n_a+n_x)
bc = np.random.randn(n_a,1)
Wy = np.random.randn(n_y,n_a)
by = np.random.randn(n_y,1)

# initialize outputs: "a" and "y" 
a = np.zeros((n_a, m, T_x))
c = np.zeros((n_a, m, T_x))
Y_pred = np.zeros((n_y, m, T_x))

# loop over all time-steps for each cell
a0 = np.random.randn(n_a,m)
a_prev = a0  # shape: (n_a,m)
c_prev = np.zeros(a_prev.shape)  # shape: (n_a,m)
for t in range(T_x):
    # Concatenate a_prev and xt
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = X[:,:,t]

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given 
    ft = sigmoid(np.dot(Wf, concat) + bf) # shape: (n_a,m)
    it = sigmoid(np.dot(Wi, concat) + bi) # shape: (n_a,m)
    cct = np.tanh(np.dot(Wc, concat) + bc) # shape: (n_a,m)
    c_next = ft * c_prev + it * cct  # shape: (n_a,m)
    ot = sigmoid(np.dot(Wo, concat) + bo) # shape: (n_a,m)
    a_next = ot * np.tanh(c_next) # shape: (n_a,m)
    
    
    # Save prediction of the LSTM cell
    Y_pred[:,:,t] = softmax(np.dot(Wy, a_next) + by) # shape: (n_y,m)
    a[:,:,t] = a_next
    c[:,:,t] = c_next
    
    # change for (t+1) round
    a_prev = a_next
    c_prev = c_next
    
    

print("a.shape = ", a.shape)
print("y_pred.shape = ", Y_pred.shape)
print("c.shape = ", c.shape)
print("Input shape ", X.shape)
print("After RNN layer, output shape ", c[:, :, 3].shape)
```

