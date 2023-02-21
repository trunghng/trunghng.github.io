---
title: "Probabilistic Graphical Models - Learning"
date: 2023-02-19T17:23:56+07:00
draft: true
tags: [machine-learning, probabilistic-graphical-model]
math: true
eqn-number: true
---
Notes on Learning in PGMs.
<!--more-->

## Maximum Likelihood Estimation{#mle}

### MLE for Bayesian Networks{#mle-bn}
Suppose that we have a Bayesian network of two binary nodes $X,Y$ connected by $X\to Y$.
<figure>
	<img width="30%" height="30%" src="/images/pgm-learning/mle-bn.png" alt="BN"/>
</figure>

The network is parameterized by a parameter vector $\boldsymbol{\theta}$, which defines the set of parameters of all the CPDs in the network, i.e. $\boldsymbol{\theta}\_X=\\{\theta_{x^0},\theta_{x^1}\\}$ and $\boldsymbol{\theta}\_{Y\vert X}=\boldsymbol{\theta}\_{Y\vert x_0}\cup\boldsymbol{\theta}\_{Y\vert x^1}=\\{\theta_{y^0\vert x^0},\theta_{y^1\vert x^0}\\}\cup\\{\theta_{y^0\vert x^1},\theta_{y^1\vert x^1}\\}$.

Assuming that we are given the training set
\begin{equation}
\mathcal{D}=\\{(x[1],y[1]),\ldots,(x[M],y[M])\\}
\end{equation}
which describes $M$ instances of variables $X$ and $Y$. The likelihood function is then given as
\begin{align}
L(\boldsymbol{\theta})&=\prod_{m=1}^{M}P(x[m],y[m];\boldsymbol{\theta}) \\\\ &=\prod_{m=1}^{M}P(x[m];\boldsymbol{\theta})P(y[m]\big\vert x[m];\boldsymbol{\theta}) \\\\ &=\left(\prod_{m=1}^{M}P(x[m];\boldsymbol{\theta})\right)\left(\prod_{m=1}^{M}P(y[m]\big\vert x[m];\boldsymbol{\theta})\right),
\end{align}
which decomposes into two terms, on for each variable. Each of these are referred as **local likelihood function** that measures how well the variable is predicted given its parents.

#### Global Likelihood Decomposition
Generally, suppose that we want to learn a parameters $\boldsymbol{\theta}$ for Bayesian network structure $\mathcal{G}$. Given a dataset $\mathcal{D}=\\{\xi[1],\ldots,\xi[M]\\}$, analogy to the argument above, we have that the likelihood function is given by
\begin{align}
L(\boldsymbol{\theta})&=\prod_{m=1}^{M}P_\mathcal{G}(\xi[m];\boldsymbol{\theta}) \\\\ &=\prod_{m=1}^{M}\prod_i P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\big) \\\\ &=\prod_i\left[\prod_{m=1}^{M}P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\big)\right]\label{eq:gld.1}
\end{align}
Each of the terms in the square brackets refers to the **conditional likelihood** of a particular variable given its parents in the network. Also, let $\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}$ denote the subset of parameters that determines $P(X_i\vert\text{Pa}\_{X_i})$. Thus, the local likelihood function for $X_i$ is then given by
\begin{equation}
L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}})=\prod_{m=1}^{M}P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}\big),
\end{equation}
which allows us to rewrite the likelihood function \eqref{eq:gld.1} as
\begin{equation}
L(\boldsymbol{\theta})=\prod_i L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}})
\end{equation}
In other words, when $\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}$ are disjoint, the likelihood can be decomposed as a product of independent terms, one for each CPD of the network. This property is known as the **global decomposition** of the likelihood function.

Additionally, we can maximize each local likelihood function $L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}})$ independently of the others, and then combine the solutions together to get an MLE solution.

#### Table-CPDs
As the MLE solution for a Bayesian network can be computed via parameterization of its CPDs, we now consider the simplest parameterization of the CPD, tabular CPD, or table-CPD.

Suppose we have a variable $X$ with parents $\mathbf{U}$. If we represent the CPD $P(X\vert\mathbf{U})$ as a table, we then have a parameter $\theta_{x\vert\mathbf{u}}$ for each $x\in\text{Val}(X)$ and $\mathbf{u}\in\text{Val}(\mathbf{U})$. The local likelihood function is then can be decomposed further as
\begin{align}
L_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}})&=\prod_{m=1}^{M}\theta_{x[m]\vert\mathbf{u}[m]} \\\\ &=\prod_{\mathbf{u}\in\text{Val}(\mathbf{U})}\left(\prod_{x\in\text{Val}(X)}\theta_{x\vert\mathbf{u}}^{M[\mathbf{u},x]}\right),
\end{align}
where $M[\mathbf{u},x]$ is the number of times $x[m]=x$ and $\mathbf{u}[m]=\mathbf{u}$ in $\mathcal{D}$.

#### Gaussian Bayesian Networks
Consider a variable $X$ with parents $\mathbf{U}=\\{U_1,\ldots,U_k\\}$ with a linear Gaussian CPD
\begin{equation}
P(X\vert\mathbf{u})=\mathcal{N}(\beta_0+\beta_1 u_1+\ldots+\beta_k u_k;\sigma^2)
\end{equation}



## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes
