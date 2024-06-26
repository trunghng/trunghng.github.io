---
title: "Neural networks"
date: 2022-09-02 13:00:00 +0700
tags: [machine-learning, neural-network]
math: true
eqn-number: true
draft: true
---
> Notes on Neural networks.
<!--more-->

## Feed-forward network functions{#ff-func}
Recall that the [linear models]({{< ref "glm" >}}) used in regression and classification are based on linear combination of fixed nonlinear basis function $\phi_j(\mathbf{x})$ and take the form
\begin{equation}
y(\mathbf{x},\mathbf{w})=f\left(\sum_{j=1}^{M}w_j\phi_j(\mathbf{x})\right),\label{1}
\end{equation}
where in the case of regression, $f$ is the function $f(x)=x$, while in the classification case, $f$ takes the form of a nonlinear activation function (e.g., the [sigmoid function]({{< ref "glm#logistic-sigmoid-func" >}})).

**Neural networks** extend this model \eqref{1} by letting each basis functions $\phi_j(\mathbf{x})$ be a nonlinear function of a linear combination of the inputs, where the coefficients in the combination are the adaptive parameters.

More formally, neural networks is a series of layers, in which each layer represents a functional transformation. Let us consider the first layer by constructing $M$ linear combinations of the input variable $x_1,\ldots,x_D$ in the form
\begin{equation}
a_j=\sum_{i=1}^{D}w_{ji}^{(1)}x_i+w_{j0}^{(1)},\label{2}
\end{equation}
where
- $j=1,\ldots,M$;
- the superscript $^{(1)}$ indicates that we are working with parameters of the first layer;
- $w_{ji}^{(1)}$'s are called the **weights**;
- $w_{j0}^{(1)}$'s are known as the **biases**;
- $a_j$'s are referred as **activations**.

The activations $a_j$'s are then transformed using a differentiable, nonlinear **activation function** $h(\cdot)$, which correspond to $f(\cdot)$ in \eqref{1} to give
\begin{equation}
z_j=h(a_j),\label{3}
\end{equation}
where $z_j$ are called the **hidden units**. Repeating the same procedure as \eqref{2}, which was following \eqref{1}, $z_j$'s are taken as the inputs of the second layer to give $K$ outputs
\begin{equation}
a_k=\sum_{j=1}^{M}w_{kj}^{(2)}z_j+w_{k0}^{(2)},\label{4}
\end{equation}
where $k=1,\ldots,K$.

This process will be repeated in $L$ times with $L$ is the number of layers. At the last layer, for instance, the second layer of our example network, the outputs, also called **output unit activations**, $a_k$'s are transformed using an appropriate activation function to give a set of network output $y_k$. For example, in multiple binary classification problems, we can choose the logistic sigmoid as our activation function that
\begin{equation}
y_k=\sigma(a_k)\label{5}
\end{equation}
Combining all these steps \eqref{2}, \eqref{3}, \eqref{4} and \eqref{5} together, our neural network with sigmoidal output unit activation functions can be defined as
\begin{equation}
y_k(\mathbf{x},\mathbf{w})=\sigma\left(\sum_{j=1}^{M}w_{kj}^{(2)}h\left(\sum_{i=1}^{D}w_{ji}^{(1)}x_i+w_{j0}^{(1)}\right)+w_{k0}^{(2)}\right),\label{6}
\end{equation}
where all of the weights and biases are comprises together into a parameter vector $\mathbf{w}$. As suggested in [linear regression]({{< ref "glm#dummy-coeff" >}}), we can also let the bias $w_{j0}^{(1)}$ be coefficient of a dummy input variable $x_0=1$ that makes \eqref{2} can be written as
\begin{equation}
a_j=\sum_{i=0}^{D}w_{ji}^{(1)}x_i
\end{equation}
This results that our subsequent layers are also able to be written in a more convenient form, which lets the entire network \eqref{6} take the form
\begin{equation}
y_k(\mathbf{x},\mathbf{w})=\sigma\left(\sum_{j=0}^{M}w_{kj}^{(2)}h\left(\sum_{i=0}^{D}w_{ji}^{(1)}x_i\right)\right)
\end{equation}
Our network is also an example of a **multilayer perception**, or **MLP**, which is a combination of [perceptron models]({{< ref "glm#perceptron" >}}). The key difference is that while the neural network uses continuous sigmoidal nonlinearities in the hidden units, which is differentiable w.r.t the parameters, the perceptron algorithm uses step-function nonlinearities, which is in contrast non-differentiable.

The network network we have been considering so far is **feed-forward neural network**, whose outputs are deterministic functions of the inputs. Each (hidden or output) unit in such a network computes a function given by
\begin{equation}
z_k=h\left(\sum_{j}w_{kj}z_j\right),
\end{equation}
where the sum runs all over units sending connections to unit $k$ (bias included).

### Universal approximation property{#unv-approx}
Feed-forward networks with **hidden layers** (i.e., the layers in which the training data does not show the desired output, e.g., the first layer of our network, the second layer on the other hands is called the **output layer**) provide **universal approximation** property.

In concrete, the universal approximation theorem states that a feedforward network with a linear output layer and at least one hidden layer with any **squashing** activation function (e.g., the logistic sigmoid function) an approximate any continuous function on a compact subsets of $\mathbb{R}^n$.

### Weight-space symmetries{#w-s-sym}

## Network training{#net-training}

### Network outputs probabilistic interpretation{#output-prob-itp}

#### Univariate regression{#univ-output}
Consider the [regression problem]({{< ref "glm#least-squares-reg" >}}) in which the target variable $t$ has Gaussian distribution with an $\mathbf{x}$ dependent mean
\begin{equation}
p(t\vert\mathbf{x},\mathbf{w})=\mathcal{N}(t\vert y(\mathbf{x},\mathbf{w}),\beta^{-1}),
\end{equation}
For the conditional distribution above, it is sufficient to take the output unit activation function to be the function $h(x)=x$, because such a network can approximate any continuous function from $\mathbf{x}$ to $y$.

Given the data set $(\mathbf{X},\mathbf{t})=\\{\mathbf{x}\_n,t_n\\}$, where $\mathbf{x}\_n$'s are i.i.d for $n=1,\ldots,N$, and where
\begin{align}
\mathbf{X}=\left[\begin{matrix}\vert&&\vert \\\\ \mathbf{x}\_1&\ldots&\mathbf{x}\_N \\\\ \vert&&\vert\end{matrix}\right],\hspace{1cm}\mathbf{t}=\left[\begin{matrix}t_1 \\\\ \vdots \\\\ t_N\end{matrix}\right]
\end{align}
The likelihood function therefore can be given by
\begin{align}
p(t\vert\mathbf{X},\mathbf{w},\beta)&=\prod_{n=1}^{N}p(t_n\vert\mathbf{x}\_n,\mathbf{w},\beta) \\\\ &=\prod_{n=1}^{N}\mathcal{N}(t_n\vert y(\mathbf{x}\_n,\mathbf{w}),\beta^{-1})
\end{align}
With a minor change as usual that taking negative natural logarithm of both sides gives us
\begin{align}
-\log p(\mathbf{t}\vert\mathbf{X},\mathbf{w},\beta)&=-\sum_{n=1}^{N}\log\mathcal{N}(t_n\vert y(\mathbf{x}\_n,\mathbf{w}),\beta^{-1}) \\\\ &=\frac{\beta}{2}\sum_{n=1}^{N}\big(y(\mathbf{x}\_n,\mathbf{w})-t_n\big)^2-\frac{N}{2}\log\beta+\frac{N}{2}\log 2\pi
\end{align}
Therefore, maximizing the likelihood function $p(\mathbf{t}\vert\mathbf{X},\mathbf{x},\beta)$ is equivalent to minimizing the sum-of-squares error function given as
\begin{equation}
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\big(y(\mathbf{x}\_n,\mathbf{w})-t_n\big)^2,
\end{equation}
This also means the value of $\mathbf{w}$ that minimizes $E(\mathbf{w})$ will be $\mathbf{w}\_\text{ML}$, which implies that the corresponding solution for $\beta$ will be given by
\begin{equation}
\frac{1}{\beta_\text{ML}}=\frac{1}{N}\sum_{n=1}^{N}\big(y(\mathbf{x}\_n,\mathbf{w}\_\text{ML})-t_n\big)^2
\end{equation}

#### Multivariate regression{#mult-output}
Similarly, we consider the multiple target variables case, in which the conditional distribution of the target therefore takes the form
\begin{equation}
p(\mathbf{t}\vert\mathbf{x},\mathbf{w},\beta)=\mathcal{N}(\mathbf{t}\vert\mathbf{y}(\mathbf{x},\mathbf{w}),\beta^{-1}\mathbf{I})
\end{equation}
Repeating the same procedure as the univariate case, maximizing likelihood function is also equivalent to minimizing the sum-of-squares error function given by
\begin{equation}
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^{N}\big\Vert\mathbf{y}(\mathbf{x}\_n,\mathbf{w})-\mathbf{t}\_n\big\Vert^2,
\end{equation}
which gives us the solution for the noise precision $\beta$ in the multivariate case as
\begin{equation}
\frac{1}{\beta_\text{ML}}=\frac{1}{NK}\sum_{n=1}^{N}\big\Vert\mathbf{y}(\mathbf{x}\_n,\mathbf{w}\_\text{ML})-\mathbf{t}\_n\big\Vert^2,
\end{equation}
where $K$ is the number of target variables.

#### Binary classification{#bi-clf}
Consider the problem of binary classification which outputs $t=1$ to denote class $\mathcal{C}\_1$ and otherwise to denote class $\mathcal{C}\_2$.

In particular, we consider a network having a single output whose activation function is a logistic sigmoid
\begin{equation}
y=\sigma(a)\doteq\frac{1}{1+\exp(-a)},
\end{equation}
which follows immediately that $0\leq y(\mathbf{x},\mathbf{w})\leq 1$.

This suggests us interpreting $y(\mathbf{x},\mathbf{w})$ as the conditional probability for class $\mathcal{C}\_1$, $p(\mathcal{C}\_1\vert\mathbf{x})$, and hence the corresponding conditional probability for class $\mathcal{C}\_2$ will be $p(\mathcal{C}\_2\vert\mathbf{x})=1-y(\mathbf{x},\mathbf{w})$. Or in other words, the conditional distribution $p(t\vert\mathbf{x},\mathbf{w})$ of targets $t$ given inputs $\mathbf{x}$ is then a Bernoulli distribution of the form
\begin{equation}
p(t\vert\mathbf{x},\mathbf{w})=y(\mathbf{x},\mathbf{w})^t\big(1-y(\mathbf{x},\mathbf{w})\big)^{1-t}
\end{equation}
If we consider a training set of $N$ independent observations as in the two regression tasks above, the likelihood function of our classification task will be given as
\begin{align}
p(\mathbf{t}\vert\mathbf{X},\mathbf{w})&=\prod_{n=1}^{N}p(t_n\vert\mathbf{x}\_n,\mathbf{w}) \\\\ &=\prod_{n=1}^{N}y(\mathbf{x}\_n,\mathbf{w})^{t_n}\big(1-y(\mathbf{x}\_n,\mathbf{w})\big)^{1-t_n}
\end{align}
Taking the negative natural logarithm of the likelihood as above gives us the cross-entropy error function
\begin{align}
E(\mathbf{w})=-\log p(\mathbf{t}\vert\mathbf{X},\mathbf{w})&=-\log\prod_{n=1}^{N}y(\mathbf{x}\_n,\mathbf{w})^{t_n}\big(1-y(\mathbf{x}\_n,\mathbf{w})\big)^{1-t_n} \\\\ &=-\sum_{n=1}^{N}t_n\log y_n+(1-t_n)\log(1-y_n),
\end{align}
where $y_n=y(\mathbf{x}\_n,\mathbf{w})$.

Moreover, consider the partial derivative of this error function w.r.t the activation $a_i$, corresponding to a particular data point $i$, we have
\begin{align}
\frac{\partial E(\mathbf{w})}{\partial a_i}&=\frac{\partial}{\partial a_i}-\sum_{n=1}^{N}t_n\log y_n+(1-t_n)\log(1-y_n) \\\\ &=-\frac{t_i}{y_i}\frac{\partial y_i}{\partial a_i}-\frac{1-t_i}{1-y_i}\frac{\partial(1-y_i)}{\partial a_i} \\\\ &=\frac{\partial y_i}{\partial a_i}\left(\frac{1-t_i}{1-y_i}-\frac{t_i}{y_i}\right) \\\\ &=y_i(1-y_i)\left(\frac{1-t_i}{1-y_i}-\frac{t_i}{y_i}\right) \\\\ &=y_i-t_i,\label{eq:bin-clf-drv-error-a}
\end{align}
where in the forth step, we have use the identity of the [derivative of sigmoid function]({{< ref "glm#sigmoid-derivative" >}}) that
\begin{equation}
\frac{d\sigma}{d a}=\sigma(1-\sigma)
\end{equation}

#### Multi-class classification{#mult-clf}
For the multi-class classification that assigns input variables to $K$ separated classes, we can use the network with $K$ outputs each of which has a logistic sigmoid activation function. Each output $t_k\in\\{0,1\\}$ for $k=1,\ldots,K$ indicates whether the input will be assigned to class $\mathcal{C}\_k$

We first consider the case that the class labels are independent given the input vector, which means the conditional distributions for class $C_k$'s will be $K$ i.i.d Bernoulli distributions, in which the conditional probability for class $\mathcal{C}\_k$ will take the form
\begin{equation}
p(\mathcal{C}\_k\vert\mathbf{x},\mathbf{w})=y_k(\mathbf{x},\mathbf{w})^{t_k}\big(1-y_k(\mathbf{x},\mathbf{w})\big)^{1-t_k}
\end{equation}
Therefore, the joint distribution of them, the conditional distribution of the target variables will be given as
\begin{align}
p(\mathbf{t}\vert\mathbf{x},\mathbf{w})&=\prod_{k=1}^{K}p(\mathcal{C}\_k\vert\mathbf{x},\mathbf{w}) \\\\ &=\prod_{k=1}^{K}y_k(\mathbf{x},\mathbf{w})^{t_k}\big(1-y_k(\mathbf{x},\mathbf{w})\big)^{1-t_k}
\end{align}
Let $\mathbf{T}$ denote the combination of all the targets $\mathbf{t}\_n$, i.e.,
\begin{equation}
\mathbf{T}=\left[\begin{matrix}-\hspace{0.15cm}\mathbf{t}\_1^\text{T}\hspace{0.15cm}- \\\\ \vdots \\\\ -\hspace{0.15cm}\mathbf{t}\_N^\text{T}\hspace{0.15cm}-\end{matrix}\right],
\end{equation}
the likelihood function therefore takes the form of
\begin{align}
p(\mathbf{T}\vert\mathbf{X},\mathbf{w})&=\prod_{n=1}^{N}p(\mathbf{t}\_n\vert\mathbf{x}\_n,\mathbf{w}) \\\\ &=\prod_{n=1}^{N}\prod_{k=1}^{K}y_k(\mathbf{x}\_n,\mathbf{w})^{t_k}\big(1-y_k(\mathbf{x}\_n,\mathbf{w})\big)^{1-t_k}\label{eq:mult-clf-llh}
\end{align}
Analogy to the binary case, taking the negative natural logarithm of the likelihood \eqref{eq:mult-clf-llh} gives us the corresponding cross-entropy error function for the multi-class case, given as
\begin{align}
E(\mathbf{w})=-\log p(\mathbf{T}\vert\mathbf{X},\mathbf{w})&=-\log\prod_{n=1}^{N}\prod_{k=1}^{K}y_k(\mathbf{x}\_n,\mathbf{w})^{t_{nk}}\big(1-y_k(\mathbf{x}\_n,\mathbf{w})\big)^{1-t_{nk}} \\\\ &=-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\log y_{nk}+(1-t_{nk})\log(1-y_{nk}),\label{eq:mult-clf-error}
\end{align}
where $y_{nk}$ is short for $y_k(\mathbf{x}\_n,\mathbf{w})$.

Similar to the binary case, consider the partial derivative of the error function \eqref{eq:mult-clf-error} w.r.t to the activation for a particular output unit $a_{ij}$, corresponding to a particular data point $i$, we have
\begin{align}
\frac{\partial E(\mathbf{w})}{\partial a_{ij}}&=\frac{\partial}{\partial a_{ij}}-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\log y_{nk}+(1-t_{nk})\log(1-y_{nk}) \\\\ &=\left(\frac{1-t_{ij}}{1-y_{ij}}-\frac{t_{ij}}{y_{ij}}\right)\frac{\partial y_{ij}}{\partial a_{ij}} \\\\ &=\left(\frac{1-t_{ij}}{1-y_{ij}}-\frac{t_{ij}}{y_{ij}}\right)y_{ij}(1-y_{ij}) \\\\ &=y_{ij}-t_{ij}\label{eq:mult-drv-error-a}
\end{align}
which takes the same form as \eqref{eq:bin-clf-drv-error-a}

On the other hands, if each input is assigned only to one of $K$ classes (mutually exclusive), the conditional distributions for class $C_k$ will be instead given as
\begin{equation}
p(\mathcal{C}\_k\vert\mathbf{x})=p(t_k=1\vert\mathbf{x})=y_k(\mathbf{x},\mathbf{w}),
\end{equation}
and thus the conditional distribution of the targets is
\begin{equation}
p(\mathbf{t}\vert\mathbf{x},\mathbf{w})=\prod_{k=1}^{K}p(t_k=1\vert\mathbf{x})^{t_k}=\prod_{k=1}^{K}y_k(\mathbf{x},\mathbf{w})^{t_k}
\end{equation}
The likelihood is therefore given as
\begin{equation}
p(\mathbf{T}\vert\mathbf{X},\mathbf{w})=\prod_{n=1}^{N}p(\mathbf{t}\_n\vert\mathbf{x}\_n,\mathbf{w})=\prod_{n=1}^{N}\prod_{k=1}^{K}y_k(\mathbf{x}\_n,\mathbf{w})^{t_{nk}},
\end{equation}
which gives us the following cross-entropy error function by taking the negative natural logarithm
\begin{align}
E(\mathbf{w})=-\log p(\mathbf{T}\vert\mathbf{X},\mathbf{w})&=-\log\prod_{n=1}^{N}\prod_{k=1}^{K}y_k(\mathbf{x},\mathbf{w})^{t_{nk}} \\\\ &=-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\log y_k(\mathbf{x}\_n,\mathbf{w})\label{eq:mult-me-clf-error}
\end{align}
As discussed in [Softmax regression]({{< ref "glm#softmax-reg" >}}), we see that the output unit activation function is given by the softmax function
\begin{equation}
y_k(\mathbf{x},\mathbf{w})=\frac{\exp\big[a_k(\mathbf{x},\mathbf{w})\big]}{\sum_{j=1}^{K}\exp\big[a_j(\mathbf{x},\mathbf{w})\big]}
\end{equation}
Taking the derivative of the error function \eqref{eq:mult-me-clf-error}  w.r.t to the activation for a particular output unit $a_{ij}$, corresponding to a particular data point $i$, we have
\begin{align}
\frac{\partial E(\mathbf{w})}{\partial a_{ij}}&=\frac{\partial}{\partial a_{ij}}-\sum_{n=1}^{N}\sum_{k=1}^{K}t_{nk}\log y_{nk} \\\\ &=-\sum_{k=1}^{K}\frac{t_{ik}}{y_{ik}}\frac{\partial y_{ik}}{\partial a_{ij}} \\\\ &=-\sum_{k=1}^{K}\frac{t_{ik}}{y_{ik}}y_{ik}(1\\{j=k\\}-y_{ij})\label{53} \\\\ &=y_{ij}\sum_{k=1}^{K}t_{ik}-\sum_{k=1}^{K}t_{ik}1\\{j=k\\} \\\\ &=y_{ij}-t_{ij}
\end{align}
where we have used the identity of the [derivative of the softmax function]({{< ref "glm#softmax-derivative" >}}) in the forth step to obtain \eqref{53}.

### Parameter optimization{#param-opt}
In training neural network to find a value of $\mathbf{w}$ to minimize the error function $E(\mathbf{w})$, we usually start with some initial value $\mathbf{w}\_0$ and iteratively update the weight vector $\mathbf{w}$, in which the weight at time step $\tau+1$ is given as
\begin{equation}
\mathbf{w}^{(t+1)}=\mathbf{w}^{(\tau)}+\Delta\mathbf{w}^{(\tau)},
\end{equation}
where $\Delta\mathbf{w}^{(\tau)}$ is some update rule.

At each time step $\tau$, there are two distinct stages:
<ul class='number-list'>
	<li>
		Stage 1 refers to evaluating the derivatives of the error function w.r.t the weights, which can be accomplished efficiently using <b>backpropagation</b> that will be discussed in the next section.
	</li>
	<li>
		Stage 2 relates to using those computed derivatives to calculate the adjustments to be made to the weights $\mathbf{w}$. <b>Gradient descent</b>, for instance, is the simplest approach in which each time step the weights take a small step in the direction of the negative gradient, as
		\begin{equation}
		\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta\nabla_\mathbf{w}E(\mathbf{w}^{(\tau)}),
		\end{equation}
		where $\eta>0$ is called the <b>learning rate</b> of the update.
	</li>
</ul>

### Backpropagation{#backprop}
In this section, we will consider the use of **backpropagation** technique to evaluate the first and second derivatives of error-functions w.r.t the weights and also the derivatives of the network outputs w.r.t the inputs.

#### Error-function derivatives{#erf-drv}
We first consider the case of evaluating the first order derivative of the error function w.r.t to the weight parameter $\mathbf{w}$.

Consider a simple linear model where the outputs $y_k$'s are linear combinations of the input variable $x_i$'s
\begin{equation}
y_k=\sum_{i}w_{ki}x_i,
\end{equation}
together with the error function, in which the error function for the $n$ data point is defined as
\begin{equation}
E_n(\mathbf{w})=\frac{1}{2}\sum_{k}(y_{nk}-t_{nk})^2,
\end{equation}
where $y_{nk}=y_k(\mathbf{x}\_n,\mathbf{w})$.

The gradient of this error function w.r.t to a weight $w_{ji}$ then can be computed by
\begin{equation}
\frac{\partial E_n}{\partial w_{ji}}=(y_{nj}-t_{nj})x_{ni}
\end{equation}
In a general feed-forward network, each unit is a weighted sum of its inputs
\begin{equation}
a_j=\sum_{i}w_{ji}z_i
\end{equation}

#### Jacobian matrix{#jacobian-mtx}

#### Hessian matrix{#hessian-mtx}

## Bayesian neural networks{#bayes-nn}

### Posterior parameter distribution{#posterior-param-dist}

## Preferences
[1] Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732). Springer New York, NY, 2006.

[2] Ian Goodfellow & Yoshua Bengio & Aaron Courville. [Deep Learning](https://www.deeplearningbook.org). MIT Press, 2016.

## Footnotes