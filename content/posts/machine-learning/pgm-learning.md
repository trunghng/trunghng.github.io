---
title: "Probabilistic Graphical Models - Learning"
date: 2023-02-19T17:23:56+07:00
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

The network is parameterized by a parameter vector $\boldsymbol{\theta}$, which defines the set of parameters of all the CPDs in the network, i.e.
\begin{equation}
\boldsymbol{\theta}\_X=\\{\theta_{x^0},\theta_{x^1}\\}
\end{equation}
and
\begin{equation}
\boldsymbol{\theta}\_{Y\vert X}=\boldsymbol{\theta}\_{Y\vert x_0}\cup\boldsymbol{\theta}\_{Y\vert x^1}=\\{\theta_{y^0\vert x^0},\theta_{y^1\vert x^0}\\}\cup\\{\theta_{y^0\vert x^1},\theta_{y^1\vert x^1}\\}
\end{equation}
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

#### Gaussian Bayesian Networks{#gaussian-bn}
Consider a variable $X$ with parents $\mathbf{U}=\\{U_1,\ldots,U_k\\}$ with a [linear Gaussian CPD]({{< ref "pgm-representation#linear-gaussian-model" >}})
\begin{equation}
P(X\vert\mathbf{u})=\mathcal{N}(\beta_0+\beta_1 u_1+\ldots+\beta_k u_k;\sigma^2)
\end{equation}
Thus, we have that
\begin{equation}
P(x\vert\mathbf{u})=\frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{(\beta_0+\beta_1 u_1+\ldots+\beta_k u_k-x)^2}{2\sigma^2}\right]
\end{equation}
Our task is to learn the parameters $\boldsymbol{\theta}\_{X\vert\mathbf{U}}=(\beta_0,\ldots,\beta_k,\sigma)$. We continue by considering the log-likelihood
\begin{align}
\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}})&=\log L_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}}) \\\\ &=\log\prod_{m=1}^{M}P\big(x[m]\big\vert\mathbf{u}[m];\boldsymbol{\theta}\_{X\vert\mathbf{U}}\big) \\\\ &=\sum_{m=1}^{M}\log P\big(x[m]\big\vert\mathbf{u}[m];\boldsymbol{\theta}\_{X\vert\mathbf{U}}\big) \\\\ &=\sum_{m=1}^{M}\left[\frac{1}{2}\log(2\pi\sigma^2)-\frac{1}{2}\frac{1}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big)^2\right]
\end{align}
Taking the derivative of the log-likelihood w.r.t $\beta_0$ gives us
\begin{align}
\frac{\partial}{\partial\beta_0}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}})&=\sum_{m=1}^{M}-\frac{1}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big) \\\\ &=-\frac{1}{\sigma^2}\left(M\beta_0+\beta_1\sum_{m=1}^{M}u_1[m]+\ldots+\beta_k\sum_{m=1}^{M}u_k[m]-\sum_{m=1}^{M}x[m]\right)
\end{align}
Setting the derivative to zero, we have that
\begin{equation}
\frac{1}{M}\sum_{m=1}^{M}x[m]=\beta_0+\beta_1\frac{1}{M}\sum_{m=1}^{M}u_1[m]+\ldots+\beta_k\frac{1}{M}\sum_{m=1}^{M}u_k[m]\label{eq:gbn.1}
\end{equation}
Then, if we define
\begin{equation}
\mathbb{E}\_\mathcal{D}[X]\doteq\frac{1}{M}\sum_{m=1}^{M}x[m],
\end{equation}
which represents the average value a variable $X$. Thus, we can rewrite \eqref{eq:gbn.1} as
\begin{equation}
\mathbb{E}\_\mathcal{D}[X]=\beta_0+\beta_1\mathbb{E}\_\mathcal{D}[U_1]+\ldots+\beta_k\mathbb{E}\_\mathcal{D}[U_k]\label{eq:gbn.2}
\end{equation}
On the other hand, differentiating the log-likelihood function w.r.t $\beta_i$ for $i\neq 0$ gives us
\begin{align}
&\hspace{-0.5cm}\frac{\partial}{\partial\beta_i}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}})\nonumber \\\\ &\hspace{-0.5cm}=\sum_{m=1}^{M}-\frac{u_i[m]}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big) \\\\ &\hspace{-0.5cm}=-\frac{1}{\sigma^2}\left(M\beta_0+\beta_1\sum_{m=1}^{M}u_1[m]u_i[m]+\ldots+\beta_k\sum_{m=1}^{M}u_k[m]u_i[m]-\sum_{m=1}^{M}x[m]u_i[m]\right)
\end{align}
Similarly, setting this derivative to zero lets us obtain
\begin{equation}
\mathbb{E}\_\mathcal{D}[X U_i]=\beta_0\mathbb{E}\_\mathcal{D}[U_i]+\beta_1\mathbb{E}\_\mathcal{D}[U_1 U_i]+\ldots+\beta_k\mathbb{E}\_\mathcal{D}[U_k U_i]\label{eq:gbn.3}
\end{equation}
From the results \eqref{eq:gbn.2} and \eqref{eq:gbn.3}, we can find the MLE solution $\hat{\boldsymbol{\beta}}$ by solving the system of linear equations
\begin{equation}
\left[\begin{matrix}1&\mathbb{E}\_\mathcal{D}[U_1]&\ldots&\mathbb{E}\_\mathcal{D}[U_k] \\\\ \mathbb{E}\_\mathcal{D}[U_1]&\mathbb{E}\_\mathcal{D}[U_1 U_1]&\ldots&\mathbb{E}\_\mathcal{D}[U_1 U_k] \\\\ \vdots&\vdots&\ddots&\vdots \\\\ \mathbb{E}\_\mathcal{D}[U_k]&\mathbb{E}\_\mathcal{D}[U_k U_1]&\ldots&\mathbb{E}\_\mathcal{D}[U_k U_k]\end{matrix}\right]\left[\begin{matrix}\beta_0 \\\\ \beta_1 \\\\ \vdots \\\\ \beta_k\end{matrix}\right]=\left[\begin{matrix}\mathbb{E}\_\mathcal{D}[X] \\\\ \mathbb{E}\_\mathcal{D}[X U_1] \\\\ \vdots \\\\ \mathbb{E}\_\mathcal{D}[X U_k]\end{matrix}\right]
\end{equation}
Additionally, by multiplying both sides of \eqref{eq:gbn.2} with $\mathbb{E}\_\mathcal{D}[U_i]$, we have
\begin{equation}
\mathbb{E}\_\mathcal{D}[X]\mathbb{E}\_\mathcal{D}[U_i]=\beta_0\mathbb{E}\_\mathcal{D}[U_i]+\beta_1\mathbb{E}\_\mathcal{D}[U_1]\mathbb{E}\_\mathcal{D}[U_i]+\ldots+\beta_k\mathbb{E}\_\mathcal{D}[U_k]\mathbb{E}\_\mathcal{D}[U_i]
\end{equation}
Then, subtracting this equation from \eqref{eq:gbn.3} gives us
\begin{align}
\mathbb{E}\_\mathcal{D}[X U_i]-\mathbb{E}\_\mathcal{D}[X]\mathbb{E}\_\mathcal{D}[U_i]&=\beta_1\big(\mathbb{E}\_\mathcal{D}[U_1 U_i]-\mathbb{E}\_\mathcal{D}[U_1]\mathbb{E}\_\mathcal{D}[U_i]\big)+\ldots+\nonumber \\\\ &\hspace{0.6cm}\beta_k\big(\mathbb{E}\_\mathcal{D}[U_k U_i]-\mathbb{E}\_\mathcal{D}[U_k]\mathbb{E}\_\mathcal{D}[U_i]\big),
\end{align}
or
\begin{equation}
\text{Cov}\_\mathcal{D}[X,U_i]=\beta_1\text{Cov}\_\mathcal{D}[U_1,U_i]+\ldots+\beta_k\text{Cov}\_\mathcal{D}[U_k,U_i]\label{eq:gbn.4}
\end{equation}
where we have defined $\text{Cov}_\mathcal{D}[X,U_i]$ as the observed covariance of $X$ and $U_i$ in the data.

Finally, differentiating the log-likelihood w.r.t $\sigma^2$, we have that
\begin{equation}
\frac{\partial}{\partial\sigma}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}})=\sum_{m=1}^{M}\left[\frac{1}{2}\frac{1}{\sigma^2}+\frac{1}{2}\frac{1}{(\sigma^2)^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big)^2\right]
\end{equation}
Analogously, setting this derivative to zero, we have that
\begin{equation}
\sigma^2=\text{Cov}\_\mathcal{D}[X,X]-\sum_{i=1}^{k}\sum_{j=1}^{k}\beta_i\beta_j\text{Cov}\_\mathcal{D}[U_i,U_j]\label{eq:gbn.5}
\end{equation}

**Remark**  


### Bayesian Parameter Estimation

#### Joint Probabilistic Model


## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes
