---
title: "Probabilistic Graphical Models - Learning"
date: 2023-02-19T17:23:56+07:00
tags: [machine-learning, probabilistic-graphical-model]
math: true
eqn-number: true
---
Notes on Learning in PGMs.
<!--more-->

## Parameter Estimation in Bayesian Networks

### MLE for Bayesian Networks{#mle-bn}
Suppose that we have a Bayesian network of two binary nodes $X,Y$ connected by $X\to Y$.
<figure>
	<img width="27%" height="27%" src="/images/pgm-learning/mle-bn.png" alt="BN"/>
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
L(\boldsymbol{\theta};\mathcal{D})&=\prod_{m=1}^{M}P(x[m],y[m];\boldsymbol{\theta}) \\\\ &=\prod_{m=1}^{M}P(x[m];\boldsymbol{\theta})P(y[m]\big\vert x[m];\boldsymbol{\theta}) \\\\ &=\left(\prod_{m=1}^{M}P(x[m];\boldsymbol{\theta})\right)\left(\prod_{m=1}^{M}P(y[m]\big\vert x[m];\boldsymbol{\theta})\right),
\end{align}
which decomposes into two terms, on for each variable. Each of these are referred as **local likelihood function** that measures how well the variable is predicted given its parents.

#### Global Likelihood Decomposition
Generally, suppose that we want to learn a parameters $\boldsymbol{\theta}$ for Bayesian network structure $\mathcal{G}$. Given a dataset $\mathcal{D}=\\{\xi[1],\ldots,\xi[M]\\}$, analogy to the argument above, we have that the likelihood function is given by
\begin{align}
L(\boldsymbol{\theta};\mathcal{D})&=\prod_{m=1}^{M}P_\mathcal{G}(\xi[m];\boldsymbol{\theta}) \\\\ &=\prod_{m=1}^{M}\prod_i P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\big) \\\\ &=\prod_i\left[\prod_{m=1}^{M}P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\big)\right]\label{eq:gld.1}
\end{align}
Each of the terms in the square brackets refers to the **conditional likelihood** of a particular variable given its parents in the network. Also, let $\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}$ denote the subset of parameters that determines $P(X_i\vert\text{Pa}\_{X_i})$. Thus, the local likelihood function for $X_i$ is then given by
\begin{equation}
L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}};\mathcal{D})=\prod_{m=1}^{M}P\big(x_i[m]\big\vert\text{pa}\_{X_i}[m];\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}\big),
\end{equation}
which allows us to rewrite the likelihood function \eqref{eq:gld.1} as
\begin{equation}
L(\boldsymbol{\theta};\mathcal{D})=\prod_i L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}};\mathcal{D})
\end{equation}
In other words, when $\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}$ are disjoint, the likelihood can be decomposed as a product of independent terms, one for each CPD of the network. This property is known as the **global decomposition** of the likelihood function.

Additionally, we can maximize each local likelihood function $L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}};\mathcal{D})$ independently of the others, and then combine the solutions together to get an MLE solution.

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
\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}};\mathcal{D})&=\log L_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}}) \\\\ &=\log\prod_{m=1}^{M}P\big(x[m]\big\vert\mathbf{u}[m];\boldsymbol{\theta}\_{X\vert\mathbf{U}}\big) \\\\ &=\sum_{m=1}^{M}\log P\big(x[m]\big\vert\mathbf{u}[m];\boldsymbol{\theta}\_{X\vert\mathbf{U}}\big) \\\\ &=\sum_{m=1}^{M}\left[\frac{1}{2}\log(2\pi\sigma^2)-\frac{1}{2}\frac{1}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big)^2\right]
\end{align}
Taking the derivative of the log-likelihood w.r.t $\beta_0$ gives us
\begin{align}
\frac{\partial}{\partial\beta_0}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}};\mathcal{D})&=\sum_{m=1}^{M}-\frac{1}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big) \\\\ &=-\frac{1}{\sigma^2}\left(M\beta_0+\beta_1\sum_{m=1}^{M}u_1[m]+\ldots+\beta_k\sum_{m=1}^{M}u_k[m]-\sum_{m=1}^{M}x[m]\right)
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
&\hspace{-0.5cm}\frac{\partial}{\partial\beta_i}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}};\mathcal{D})\nonumber \\\\ &\hspace{-0.5cm}=\sum_{m=1}^{M}-\frac{u_i[m]}{\sigma^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big) \\\\ &\hspace{-0.5cm}=-\frac{1}{\sigma^2}\left(M\beta_0+\beta_1\sum_{m=1}^{M}u_1[m]u_i[m]+\ldots+\beta_k\sum_{m=1}^{M}u_k[m]u_i[m]-\sum_{m=1}^{M}x[m]u_i[m]\right)
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
\frac{\partial}{\partial\sigma}\ell_X(\boldsymbol{\theta}\_{X\vert\mathbf{U}};\mathcal{D})=\sum_{m=1}^{M}\left[\frac{1}{2}\frac{1}{\sigma^2}+\frac{1}{2}\frac{1}{(\sigma^2)^2}\big(\beta_0+\beta_1 u_1[m]+\ldots+\beta_k u_k[m]-x[m]\big)^2\right]
\end{equation}
Analogously, setting this derivative to zero, we have that
\begin{equation}
\sigma^2=\text{Cov}\_\mathcal{D}[X,X]-\sum_{i=1}^{k}\sum_{j=1}^{k}\beta_i\beta_j\text{Cov}\_\mathcal{D}[U_i,U_j]\label{eq:gbn.5}
\end{equation}

**Remark**  


**Example 1**: Let us estimate a joint multivariate Gaussian distribution. Specifically, consider continuous r.v.s $X$ and $Y$ and assume we have a dataset of $M$ samples $\mathcal{D}=\\{(x[1],y[1]),\ldots,(x[M],y[M])\\}$. Our job is to find the MLE estimate for a joint Gaussian distribution over $X,Y$.

Let $\mathbf{Z}$ be a random vector that encodes the joint distribution of $X$ and $Y$. In particular
\begin{equation}
\mathbf{Z}=\left[\begin{matrix}X \\\\ Y\end{matrix}\right]
\end{equation}
We have that $\mathbf{Z}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ with
\begin{equation}
\boldsymbol{\mu}=\left[\begin{matrix}\mu_X \\\\ \mu_Y\end{matrix}\right],\hspace{1cm}\boldsymbol{\Sigma}=\left[\begin{matrix}\Sigma_{XX}&\Sigma_{XY} \\\\ \Sigma_{YX}&\Sigma_{YY}\end{matrix}\right]
\end{equation}
Thus, we have that
\begin{equation}
P(\mathbf{z})=\frac{1}{2\pi\vert\boldsymbol{\Sigma}\vert^{1/2}}\exp\left[-\frac{1}{2}(\mathbf{z}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{z}-\boldsymbol{\mu})\right]
\end{equation}
Our job then is to learn the parameter $\boldsymbol{\theta}=(\boldsymbol{\mu},\boldsymbol{\Sigma})$. We begin by considering the log-likelihood function
\begin{align}
\ell(\boldsymbol{\theta})&=\log\prod_{m=1}^{M}P(x[m],y[m];\boldsymbol{\theta}) \\\\ &=\log\prod_{m=1}^{M}\frac{1}{2\pi\vert\boldsymbol{\Sigma}\vert^{1/2}}\exp\left[-\frac{1}{2}(\mathbf{z}[m]-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{z}[m]-\boldsymbol{\mu})\right] \\\\ &=\sum_{m=1}^{M}-\log(2\pi)-\frac{1}{2}\log\vert\boldsymbol{\Sigma}\vert-\frac{1}{2}(\mathbf{z}[m]-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{z}[m]-\boldsymbol{\mu})
\end{align}
Taking the derivative of the log-likelihood w.r.t $\boldsymbol{\mu}$ gives us
\begin{align}
\frac{\partial}{\partial\boldsymbol{\mu}}\ell(\boldsymbol{\theta})&=\sum_{m=1}^{M}\frac{1}{2}\big[\boldsymbol{\Sigma}^{-1}(\mathbf{z}[m]-\boldsymbol{\mu})-(\boldsymbol{\Sigma}^{-1})^\text{T}(\mathbf{z}[m]-\boldsymbol{\mu})\big] \\\\ &=\sum_{m=1}^{M}\boldsymbol{\Sigma}^{-1}(\mathbf{z}[m]-\boldsymbol{\mu}),
\end{align}
where we have used the fact that the covariance matrix $\boldsymbol{\Sigma}$ is symmetric. Setting the derivative to zero we obtain the MLE solution for $\boldsymbol{\mu}$
\begin{equation}
\boldsymbol{\mu}=\frac{1}{M}\sum_{m=1}^{M}\mathbf{z}[m]=\left[\begin{matrix}\mathbb{E}\_\mathcal{D}[X] \\\\ \mathbb{E}\_\mathcal{D}[Y]\end{matrix}\right]
\end{equation}
On the other hand, differentiating the log-likelihood w.r.t $\boldsymbol{\Sigma}$, we obtain
\begin{align}
\frac{\partial}{\partial\boldsymbol{\Sigma}}\ell(\boldsymbol{\theta})&=\sum_{m=1}^{M}-\frac{1}{2}\frac{\vert\boldsymbol{\Sigma}\vert\boldsymbol{\Sigma}^{-1}}{\vert\boldsymbol{\Sigma}\vert}+\frac{1}{2}\big[(\boldsymbol{\Sigma}^{-1})^\text{T}(\mathbf{z}[m]-\boldsymbol{\mu})(\mathbf{z}[m]-\boldsymbol{\mu})^\text{T}(\boldsymbol{\Sigma}^{-1})^\text{T}\big] \\\\ &=\frac{1}{2}\sum_{m=1}^{M}\boldsymbol{\Sigma}^{-1}(\mathbf{z}[m]-\boldsymbol{\mu})(\mathbf{z}[m]-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}-\boldsymbol{\Sigma}^{-1}
\end{align}
Setting this derivative to zero, we have that
\begin{align}
\boldsymbol{\Sigma}&=\sum_{m=1}^{M}(\mathbf{z}[m]-\boldsymbol{\mu})(\mathbf{z}[m]-\boldsymbol{\mu})^\text{T} \\\\ &=\sum_{m=1}^{M}\left[\begin{matrix}(x[m]-\mu_X)^2&(x[m]-\mu_X)(y[m]-\mu_Y) \\\\ (y[m]-\mu_Y)(x[m]-\mu_X)&(y[m]-\mu_Y)^2\end{matrix}\right] \\\\ &=\left[\begin{matrix}\text{Cov}\_\mathcal{D}[X,X]&\text{Cov}\_\mathcal{D}[X,Y] \\\\ \text{Cov}\_\mathcal{D}[Y,X]&\text{Cov}\_\mathcal{D}[Y,Y]\end{matrix}\right]
\end{align}

#### MLE as Moment Projection

##### M-Projections
Let $P$ be a distribution and let $\mathcal{Q}$ be a convex set of distributions. The **M-projection** (for **moment projection**) of $P$ on $\mathcal{Q}$ is the distribution
\begin{equation}
Q^M=\underset{Q\in\mathcal{Q}}{\text{argmin}}D_\text{KL}(P\Vert Q)
\end{equation}

**Theorem 1**: *Let $P$ be a distribution over $X_1,\ldots,X_n$, and let $\mathcal{Q}$ be the family of distributions consistent with $\mathcal{G}_\emptyset$. Then*
\begin{equation}
Q^M=\underset{Q\models\mathcal{G}\_\emptyset}{\text{argmin}}D_\text{KL}(P\Vert Q)
\end{equation}
*is the distribution*
\begin{equation}
Q^M(X_1,\ldots,X_n)=P(X_1)\ldots P(X_n)
\end{equation}

**Proof**  
Consider a distribution $Q\models\mathcal{G}\_\emptyset$. Since $Q$ factorizes, we have that
\begin{align}
D_\text{KL}(P\Vert Q)&=\mathbb{E}\_P\big[\log P(X_1,\ldots,X_n)-\log Q(X_1.\ldots,X_n)\big] \\\\ &=\mathbb{E}\_P\left[P(X_1,\ldots,X_n)-\sum_i\mathbb{E}\_P\big[\log Q(X_i)\big]\right] \\\\ &=\mathbb{E}\_P\left[\log\frac{P(X_1,\ldots,X_n)}{P(X_1)\ldots P(X_n)}\right]\sum_i\mathbb{E}\_P\left[\log\frac{P(X_i)}{Q(X_i)}\right] \\\\ &=D_\text{KL}(P\Vert Q^M)+\sum_i D_\text{KL}\big(P(X_i)\Vert Q(X_i)\big) \\\\ &\geq D_\text{KL}(P\Vert Q^M)
\end{align}


##### MLE as M-Projection
<b id='lm2'>Lemma 2</b>: *For any distribution $P,P'$ over $\mathcal{X}$, we have*
\begin{equation}
D_\text{KL}(P\Vert P')=-H_P(\mathcal{X})-\mathbb{E}\_{\xi\sim P}\big[\log P'(\xi)\big]
\end{equation}

**Proof**  
We have that the relative entropy is given as
\begin{align}
D_\text{KL}(P\Vert P')&=\mathbb{E}\_{\xi\sim P}\left[\log\left(\frac{P(\xi)}{P'(\xi)}\right)\right] \\\\ &=\mathbb{E}\_{\xi\sim P}\big[\log P(\xi)\big]-\mathbb{E}\_{\xi\sim P}\big[\log P'(\xi)\big] \\\\ &=-H_P(\mathcal{X})-\mathbb{E}\_{\xi\sim P}\big[\log P'(\xi)\big]
\end{align}

**Proposition 3**: *Let $\mathcal{D}$ be a dataset, then*
\begin{equation}
\log L(\boldsymbol{\theta};\mathcal{D})=M\cdot\mathbb{E}\_{\hat{P}\_\mathcal{D}}\big[\log P(\mathcal{X};\boldsymbol{\theta})\big],
\end{equation}
*where $\hat{P}_\mathcal{D}$ is the empirical distribution, defined as*
\begin{equation}
\hat{P}\_\mathcal{D}(A)=\frac{1}{M}\sum_{m=1}^{M}\mathbf{1}\\{\xi[m]\in A\\}
\end{equation}

**Proof**  
We have that
\begin{align}
\log L(\boldsymbol{\theta};\mathcal{D})&=\log\prod_{m=1}^{M}P(\xi[m];\boldsymbol{\theta}) \\\\ &=\sum_{m=1}^{M}\log P(\xi[m];\boldsymbol{\theta}) \\\\ &=\sum_\xi\left(\sum_{m=1}^{M}\mathbf{1}\\{\xi[m]=\xi\\}\right)\log P(\xi;\boldsymbol{\theta}) \\\\ &=\sum_\xi M\cdot\hat{P}\_\mathcal{D}(\xi)\log P(\xi;\boldsymbol{\theta}) \\\\ &=M\cdot\mathbb{E}\_{\hat{P}\_\mathcal{D}}\big[\log P(\mathcal{X};\boldsymbol{\theta})\big]
\end{align}

Applying [Lemma 2](#lm2) to this result, we have
\begin{align}
\ell(\boldsymbol{\theta};\mathcal{D})&=M\cdot\mathbb{E}\_{\hat{P}\_\mathcal{D}}\big[\log P(\mathcal{X};\boldsymbol{\theta})\big] \\\\ &=M\cdot\left[H_{\hat{P}\_\mathcal{D}}(\mathcal{X})-D_\text{KL}\big(\hat{P}\_\mathcal{D}(\mathcal{X})\Vert P(\mathcal{X};\boldsymbol{\theta})\big)\right]
\end{align}

From this result, we can derive the following relationship between MLE and M-projections

**Theorem 4**: *The MLE $\hat{\boldsymbol{\theta}}$ in a parametric family relative to a dataset $\mathcal{D}$ is the M-projection of $\hat{P}_\mathcal{D}$ onto the parametric family*
\begin{equation}
\hat{\boldsymbol{\theta}}=\underset{\boldsymbol{\theta}\in\Theta}{\text{argmin}}D_\text{KL}(\hat{P}\_\mathcal{D}\Vert P_\boldsymbol{\theta})
\end{equation}

### Bayesian Parameter Estimation

#### General setting
In the **Bayesian approach**, as before, we assume a general learning problem where we observe a training set $\mathcal{D}=\\{\xi[1],\ldots,\xi[M]\\}$ and a parametric model $P(\xi\vert\boldsymbol{\theta})$ where we can choose parameters from a parameter space $\Theta$, i.e. in this case, samples according to the probabilistic model are conditionally i.i.d given $\boldsymbol{\theta}$ instead of, recalling that in MLE, being (unconditionally) i.i.d.

##### Priors, Posteriors
To perform the task, we need to define a joint distribution $P(\mathcal{D},\boldsymbol{\theta})$ over the data and the parameters, which can be written by
\begin{equation}
P(\mathcal{D},\boldsymbol{\theta})=P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta}),
\end{equation}
where
<ul id='number-list'>
	<li>
		$P(\mathcal{D}\vert\boldsymbol{\theta})$ is the <b>likelihood function</b>, which is the probability of the observations given the parameters, as in the MLE approach.
	</li>
	<li>
		$P(\boldsymbol{\theta})$ is referred as the <b>prior distribution</b>, which encodes our prior beliefs, i.e. before data is observed. 
	</li>
</ul>

By Bayes' rule, from the likelihood and prior, combined with the defined joint distribution, we can derive the so-called **posterior distribution** over the parameters, which corresponds to our beliefs after observing the data, as
\begin{align}
P(\boldsymbol{\theta}\vert\mathcal{D})&=\frac{P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal{D})} \\\\ &=\frac{P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta})}{\int_\Theta P(\mathcal{D}\vert\boldsymbol{\theta}')P(\boldsymbol{\theta}')d\boldsymbol{\theta}'},
\end{align}
where, since the denominator is just a normalizing constant, can be expressed as
\begin{equation}
P(\boldsymbol{\theta}\vert\mathcal{D})\propto P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta})
\end{equation}
Since the posterior is a (normalized) product of the prior and the likelihood, it seems natural to require that the prior also have a form similar to the likelihood, such priors are referred as **conjugate priors**.

More formally, a family of priors $P(\boldsymbol{\theta}:\boldsymbol{\alpha})$ is **conjugate** to a particular model $P(\xi\vert\boldsymbol{\theta})$ if for any possible dataset $\mathcal{D}$ of i.i.d samples from $P(\xi\vert\boldsymbol{\theta})$, and any choice of legal hyperparameters $\boldsymbol{\alpha}$ for the prior over $\boldsymbol{\theta}$, there are hyperparameters $\boldsymbol{\alpha}'$ that describe the posterior, i.e.
\begin{equation}
P(\boldsymbol{\theta}:\boldsymbol{\alpha}')\propto P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta}:\boldsymbol{\alpha})
\end{equation}

##### Bayesian Estimator
From the posterior, we can predict the probability of future samples. Specifically, suppose that we are about to sample a new instance $\xi[M+1]$, then the **Bayesian estimator**, or the **predictive distribution**, is the posterior distribution over a new example.
\begin{align}
P(\xi[M+1]\vert\mathcal{D})&=\int P(\xi[M+1]\vert\mathcal{D},\boldsymbol{\theta})P(\boldsymbol{\theta}\vert\mathcal{D})d\boldsymbol{\theta} \\\\ &=\int P(\xi[M+1]\vert\boldsymbol{\theta})P(\boldsymbol{\theta}\vert\mathcal{D})d\boldsymbol{\theta} \\\\ &=\mathbb{E}\_{P(\boldsymbol{\theta}\vert\mathcal{D})}\big[P(\xi[M+1]\vert\boldsymbol{\theta})\big],
\end{align}
where in the second step, we use the fact that samples are i.i.d given $\boldsymbol{\theta}$.

#### The full Bayesian approach

##### Bayesian Parameter Estimation in Bayesian Networks

###### Parameter Independence and Global Decomposition
Suppose that we want to estimate parameters for a networks consisting of two variables $X$ and $Y$ with edge $X\rightarrow Y$. Also, we are given a training dataset $\mathcal{D}=\\{(x[1],y[1]),\ldots,(x[M],y[M])\\}$. Additionally, we have unknown parameters $\boldsymbol{\theta}\_X,\boldsymbol{\theta}_{Y\vert X}$. The dependencies between variables are described in the following network.
<figure id='fig1'>
	<img width="70%" height="70%" src="/images/pgm-learning/meta-network.png" alt="Normal distribution"/>
	<figcaption><b>Figure 1</b>: (taken from <a href='#pgm-book'>PGM book</a>) <b>Meta-network for i.i.d samples from a network $X\rightarrow Y$ with global parameter independence.</b> (a) <a href='{{< ref "pgm-representation#plate-models">}}'>Plate model</a>; (b) <a href='{{< ref "pgm-representation#gound-bn-plate-models">}}'>Ground Bayesian network</a></figcaption>
</figure>

We have already known that in using Bayesian approach, the samples are independent given the parameters, i.e.
\begin{equation}
\text{d-sep}(\\{X[m],Y[m]\\};\\{X[m'],Y[m']\\}\vert\\{\boldsymbol{\theta}\_X,\boldsymbol{\theta}\_{Y\vert X}\\}),
\end{equation}
which can be justified by an examination of active trails from the above figure. Moreover, the network structure given in [Figure 1](#fig1) also satisfies the **global parameter independence**.

Let $\mathcal{G}$ be a BN structure with parameters $\boldsymbol{\theta}=(\boldsymbol{\theta}\_{X_1}\vert\text{Pa}\_{X_1},\ldots,\boldsymbol{\theta}\_{X_n}\vert\text{Pa}\_{X_n})$. Then, a prior $P(\boldsymbol{\theta})$ is said to satisfy **global parameter independence** if it has the form
\begin{equation}
P(\boldsymbol{\theta})=\prod_{i=1}^{n}P(\boldsymbol{\theta}\_{X_i}\vert\text{Pa}\_{X_i})
\end{equation}
This property allows us to conclude that complete data, $\mathcal{D}$, d-separates the parameters for different CPDs, i.e.
\begin{equation}
\text{d-sep}\big(\boldsymbol{\theta}\_X;\boldsymbol{\theta}\_{Y\vert X}\big\vert\mathcal{D}\big)
\end{equation}
This is due to for each $m$, we have that any path between $X[m]$ and $Y[m]$ has the form
\begin{equation}
\boldsymbol{\theta}\_X\rightarrow X[m]\rightarrow Y[m]\leftarrow\boldsymbol{\theta}\_{Y\vert X},
\end{equation}
which is inactive given the observation $x[m],y[m]$. Thus, we obtain that
\begin{equation}
P(\boldsymbol{\theta}\_X,\boldsymbol{\theta}\_{Y\vert X}\vert\mathcal{D})=P(\boldsymbol{\theta}\_X\vert\mathcal{D})P(\boldsymbol{\theta}\_{Y\vert X}\vert\mathcal{D})
\end{equation}
This decomposition suggests that given $\mathcal{D}$, we can find the solution for the posterior over $\boldsymbol{\theta}$ by independently finding the result corresponding to the posterior over each of $\boldsymbol{\theta}_X$ and $\boldsymbol{\theta}_{Y\vert X}$, which is analogous to the [global likelihood decomposition](#global-likelihood-decomposition) in the MLE. In Bayesian setting this property has additional importance.

Generally, suppose that we are given a Bayesian network graph $\mathcal{G}$ with parameters $\boldsymbol{\theta}$. As mentioned before that in using Bayesian approach, we need to specify a prior distribution over the parameter space $P(\boldsymbol{\theta})$ and a posterior distribution over the parameters given the samples $\mathcal{D}$
\begin{equation}
P(\boldsymbol{\theta}\vert\mathcal{D})=\frac{P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal{D})}\label{eq:pigd.1}
\end{equation}
Moreover, assuming that we have global parameter independence, then combining with the global likelihood decomposition mentioned above, the posterior in \eqref{eq:pigd.1} can be continued to derive as a product of local terms:
\begin{align}
P(\boldsymbol{\theta}\vert\mathcal{D})&=\frac{P(\mathcal{D}\vert\boldsymbol{\theta})P(\boldsymbol{\theta})}{P(\mathcal{D})} \\\\ &=\frac{1}{P(\mathcal{D})}\left(\prod_i L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}};\mathcal{D})\right)\left(\prod_i P(\boldsymbol{\theta}\_{X\_i\vert\text{Pa}\_{X_i}})\right) \\\\ &=\frac{1}{P(\mathcal{D})}\prod_i\left(L_i(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}};\mathcal{D})P(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}})\right)
\end{align}
This gives rise to the following result.

**Proposition 3**: Let $\mathcal{D}$ be a complete dataset for $\mathcal{X}$, let $\mathcal{G}$ be a BN graph over $\mathcal{X}$. If $P(\boldsymbol{\theta})$ satisfies global parameter independence, then
\begin{equation}
P(\boldsymbol{\theta}\vert\mathcal{D})=\prod_i P(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}\vert\mathcal{D})
\end{equation}

**Example 2**: With the network specified in [Figure 1](#fig1), let us compute the predictive distribution. We have that
\begin{align}
&\hspace{-0.5cm}P(x[M+1],y[M+1]\vert\mathcal{D})\nonumber \\\\ &\hspace{-0.3cm}=\int P(x[M+1],y[M+1]\vert\mathcal{D},\boldsymbol{\theta})P(\boldsymbol{\theta}\vert\mathcal{D})d\boldsymbol{\theta} \\\\ &\hspace{-0.3cm}=\int P(x[M+1],y[M+1]\vert\boldsymbol{\theta})P(\boldsymbol{\theta}\vert\mathcal{D})d\boldsymbol{\theta} \\\\ &\hspace{-0.3cm}=\int P(x[M+1]\vert\boldsymbol{\theta}\_X)P(y[M+1]\vert x[M+1],\boldsymbol{\theta}\_{Y\vert X})P(\boldsymbol{\theta}\_X\vert\mathcal{D})P(\boldsymbol{\theta}\_{Y\vert X}\vert\mathcal{D})d\boldsymbol{\theta} \\\\ &\hspace{-0.3cm}=\int\int P(x[M+1]\vert\boldsymbol{\theta}\_X)P(y[M+1]\vert x[M+1],\boldsymbol{\theta}\_{Y\vert X})P(\boldsymbol{\theta}\_X\vert\mathcal{D})P(\boldsymbol{\theta}\_{Y\vert X}\vert\mathcal{D})d\boldsymbol{\theta}\_X d\boldsymbol{\theta}\_{Y\vert X} \\\\ &\hspace{-0.3cm}=\left(\int P(x[M+1]\vert\boldsymbol{\theta}\_X)P(\boldsymbol{\theta}\_X\vert\mathcal{D})d\boldsymbol{\theta}\_X\right)\nonumber \\\\ &\hspace{2cm}\left(\int P(y[M+1]\vert x[M+1],\boldsymbol{\theta}\_{Y\vert X})P(\boldsymbol{\theta}\_{Y\vert X}\vert\mathcal{D})d\boldsymbol{\theta}\_{Y\vert X}\right)
\end{align}
Thus, we can solve the prediction problem for the two variables $X$ and $Y$ independently.

In general, we can solve the prediction problem for each CPD, $P(X_i\vert\text{Pa}\_{X_i})$, independently the combine the results together, i.e.
\begin{align}
&P(X_1[M+1],\ldots,X_n[M+1]\vert\mathcal{D})\nonumber \\\\ &=\prod_{i=1}^{n}\int P\big(X_i[M+1]\big\vert\text{Pa}\_{X_i}[M+1],\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}\big)P\big(\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}\big\vert\mathcal{D}\big)d\boldsymbol{\theta}\_{X_i\vert\text{Pa}\_{X_i}}
\end{align}

##### Local Decomposition
By the global decomposition, our attention is then to solve localized Bayesian estimation problems.


#### MAP Estimation


## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

[2] Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732). Springer New York, NY, 2006.

## Footnotes
