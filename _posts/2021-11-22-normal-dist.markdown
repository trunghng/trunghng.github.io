---
layout: post
title:  "Gaussian Distribution"
date:   2021-11-22 14:46:00 +0700
tags: mathematics probability-statistics normal-distribution
description: Gaussian Distribution
comments: true
eqn-number: true
---
> The **Gaussian (Normal) distribution** is a continuous distribution with a bell-shaped PDF used widely in statistics due to the **Central Limit theorem**. The theorem states that under very weak assumptions, the sum of a large number of i.i.d. random variables has an approximately Normal distribution, regardless of the distribution of the individual r.v.s. This means we can start with independent r.v.s from almost any distribution, discrete or continuous, but once we add up a bunch of them, the distribution of the resulting random variable looks like a Gaussian distribution.
<!-- excerpt-end -->

- [Gaussian (Normal) distribution](#gauss-dist)
	- [Standard Normal](#std-norm)
- [Multivariate Normal distribution](#mvn)
	- [Bivariate Normal](#bvn)
- [Properties of the covariance matrix](#prop-cov)
	- [Symmetric](#sym-cov)
	- [Real eigenvalues](#re-cov)
	- [Projection onto eigenvectors](#proj-ev-cov)
- [Geometrical interpretation](#geo-int)
- [Conditional Gaussian distribution](#cond-gauss-dist)
- [Marginal Gaussian distribution](#marg-gauss-dist)
- [Bayes' theorem for Gaussian variables](#bayes-theorem-gauss)
- [References](#references)
- [Footnotes](#footnotes)  

$\newcommand{\Var}{\mathrm{Var}}$
$\newcommand{\Cov}{\mathrm{Cov}}$
## Gaussian (Normal) Distribution
{: #gauss-dist}
A random variable $X$ is said to be **Gaussian** or to have the **Normal distribution** with mean $\mu$ and variance $\sigma^2$ if its probability density function (PDF) is
\begin{equation}
f_X(x)=\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)
\end{equation}
which we denote as $X\sim\mathcal{N}(\mu,\sigma)$.

### Standard Normal
{: #std-normal}
When $X$ is normally distributed with mean $\mu=0$ and variance $\sigma^2=1$, we call its distribution **Standard Normal**.
\begin{equation}
X\sim\mathcal{N}(0,1)
\end{equation}
In this case, $X$ has special notations to denote its PDF and CDF, which are
\begin{equation}
\varphi(x)=\dfrac{1}{\sqrt{2\pi}}e^{-z^2/2},
\end{equation}
and
\begin{equation}
\Phi(x)=\int_{-\infty}^{x}\varphi(t)\,dt=\int_{-\infty}^{x}\dfrac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt
\end{equation}
Below are some illustrations of Normal distribution.
<figure>
	<img src="/assets/images/2021-11-22/normal.png" alt="normal distribution" style="display: block; margin-left: auto; margin-right: auto; width:  900px; height: 380px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: 10K normally distributed data points (5K each plot) were plotted as vertical bars on x-axis. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/bayes-optimization/gauss-dist.py)</span></figcaption>
</figure><br/>

## Multivariate Normal Distribution
{: #mvn}
A $k$-dimensional random vector $\mathbf{X}=\left(X_1,\dots,X_D\right)^\text{T}$ is said to have a **Multivariate Normal (MVN)** distribution if every linear combination of the $X_i$ has a Normal distribution. Which means
\begin{equation}
t_1X_1+\ldots+t_DX_D
\end{equation}
is normally distributed for any choice of constants $t_1,\dots,t_D$. Distribution of $\mathbf{X}$ then can be written in the following notation
\begin{equation}
\mathbf{X}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})
\end{equation}
where
\begin{equation}
\boldsymbol{\mu}=\mathbb{E}\mathbf{X}=\mathbb{E}\left(\mu_1,\ldots,\mu_k\right)^\text{T}=\left(\mathbb{E}X_1,\ldots,\mathbb{E}X_k\right)^\text{T}
\end{equation}
is the $D$-dimensional mean vector, and covariance matrix $\mathbf{\Sigma}\in\mathbb{R}^{D\times D}$ with
\begin{equation}
\boldsymbol{\Sigma}\_{ij}=\mathbb{E}\left(X_i-\mu_i\right)\left(X_j-\mu_j\right)=\Cov(X_i,X_j)\label{eq:mvn.1}
\end{equation}
We also have that $\boldsymbol{\Sigma}\geq 0$ (positive semi-definite matrix)[^1].

Thus, the PDF of an MVN is defined as
\begin{equation}
f_\mathbf{X}(x_1,\ldots,x_D)=\dfrac{1}{(2\pi)^{D/2}\vert\mathbf{\Sigma}\vert^{1/2}}\exp\left[-\dfrac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]\label{eq:mvn.2}
\end{equation}
With this idea, *Standard Normal* distribution in multi-dimensional case can be defined as a Gaussian with mean $\boldsymbol{\mu}=0$ (here $0$ is an $D$-dimensional vector) and identity covariance matrix $\boldsymbol{\Sigma}=\mathbf{I}\_{D\times D}$.

### Bivariate Normal
{: #bvn}
When the number of dimensions in $\mathbf{X}$, $D=2$, this special case of MVN is called the **Bivariate Normal (BVN)**.

An example of an BVN, $\mathcal{N}\left(\left[\begin{smallmatrix}0\\\\0\end{smallmatrix}\right],\left[\begin{smallmatrix}1&0.5\\\\0.8&1\end{smallmatrix}\right]\right)$, is shown as following.
<figure>
	<img src="/assets/images/2021-11-22/bvn.png" alt="monte carlo method" style="display: block; margin-left: auto; margin-right: auto; width: 750px; height: 350px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The PDF of $\mathcal{N}\left(\left[\begin{smallmatrix}0\\0\end{smallmatrix}\right],\left[\begin{smallmatrix}1&0.5\\0.8&1\end{smallmatrix}\right]\right)$. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/bayes-optimization/mvn.py)</span></figcaption>
</figure><br/>

## Properties of the covariance matrix
{: #prop-cov}

### Symmetric
{: #sym-cov}
With the definition \eqref{eq:mvn.1} of the covariance matrix $\boldsymbol{\Sigma}$, we can easily see that it is symmetric. However, notice that in the illustration of BVN, we gave the distribution a non-symmetric covariance matrix. The reason why we could do that is without loss of generality, we can assume that $\boldsymbol{\Sigma}$ is symmetric.

To prove this property, first off consider a square matrix $\mathbf{S}$, we have it can be written by
\begin{equation}
\mathbf{S}=\frac{\mathbf{S}+\mathbf{S}^\text{T}}{2}+\frac{\mathbf{S}-\mathbf{S}^\text{T}}{2}=\mathbf{S}\_\text{S}+\mathbf{S}\_\text{A},
\end{equation}
where
\begin{equation}
\mathbf{S}\_\text{S}=\frac{\mathbf{S}+\mathbf{S}^\text{T}}{2},\hspace{2cm}\mathbf{S}\_\text{A}=\frac{\mathbf{S}-\mathbf{S}^\text{T}}{2}
\end{equation}
It is easily seen that $\mathbf{S}\_\text{S}$ is symmetric because the $\\{i,j\\}$ element of its equal to the $\\{j,i\\}$ element due to
\begin{equation}
(\mathbf{S}\_\text{S})\_{ij}=\frac{(\mathbf{S})\_{ij}+(\mathbf{S}^\text{T})\_{ij}}{2}=\frac{(\mathbf{S}^\text{T})\_{ji}+(\mathbf{S})\_{ji}}{2}=(\mathbf{S}\_\text{S})\_{ji}
\end{equation}
On the other hand, the matrix $\mathbf{S}\_\text{A}$ is anti-symmetric since
\begin{equation}
(\mathbf{S}\_\text{A})\_{ij}=\frac{(\mathbf{S})\_{ij}-(\mathbf{S}^\text{T})\_{ij}}{2}=\frac{(\mathbf{S}^\text{T})\_{ji}-(\mathbf{S})\_{ji}}{2}=-(\mathbf{S}\_\text{A})\_{ji}
\end{equation}
Consider the density of a distribution $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$, we have that $\boldsymbol{\Sigma}$ is square and so is its inverse $\boldsymbol{\Sigma}^{-1}$. Therefore we can express $\boldsymbol{\Sigma}^{-1}$ as a sum of a symmetric matrix $\boldsymbol{\Sigma}\_\text{S}$ with an anti-symmetric matrix $\boldsymbol{\Sigma}\_\text{A}$
\begin{equation}
\boldsymbol{\Sigma}^{-1}=\boldsymbol{\Sigma}\_\text{S}+\boldsymbol{\Sigma}\_\text{A}
\end{equation}
We have that the density of the distribution is given by
\begin{align}
f(\mathbf{x})&=\frac{1}{(2\pi)^{D/2}\vert\boldsymbol{\Sigma}\vert^{1/2}}\exp\left[-\dfrac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\text{T}\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right] \\\\ &\propto\exp\left[-\dfrac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\text{T}\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right] \\\\ &=\exp\left[-\dfrac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\text{T}(\boldsymbol{\Sigma}\_\text{S}+\boldsymbol{\Sigma}\_\text{A})(\mathbf{x}-\boldsymbol{\mu})\right] \\\\ &\propto\exp\left[\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{S}\mathbf{v}+\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{A}\mathbf{v}\right] \\\\ &=\exp\left[\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{S}\mathbf{v}\right]
\end{align}
where in the forth step, we have defined $\mathbf{v}\doteq\mathbf{x}-\boldsymbol{\mu}$, and where in the fifth-step, the result obtained was due to
\begin{align}
\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{A}\mathbf{v}&=\sum_{i=1}^{D}\sum_{j=1}^{D}\mathbf{v}\_i(\boldsymbol{\Sigma}\_\text{A})\_{ij}\mathbf{v}\_j \\\\ &=\sum_{i=1}^{D}\sum_{j=1}^{D}\mathbf{v}\_i-(\boldsymbol{\Sigma}\_\text{A})\_{ji}\mathbf{v}\_j \\\\ &=-\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{A}\mathbf{v}
\end{align}
which implies that $\mathbf{v}^\text{T}\boldsymbol{\Sigma}\_\text{A}\mathbf{v}=0$.

Thus, when computing the density, the symmetric part of $\boldsymbol{\Sigma}^{-1}$ is the only one matters. Or in other words, without loss of generality, we can assume that $\boldsymbol{\Sigma}^{-1}$ is symmetric, which means that $\boldsymbol{\Sigma}$ is also symmetric.

With this assumption of symmetry, the covariance matrix $\boldsymbol{\Sigma}$ now has all the properties of a symmetric matrix, as following in the next two sections.

### Real eigenvalues
{: #re-cov}
Consider an eigenvector, eigenvalue pair $(\mathbf{v},\lambda)$ of covariance matrix $\boldsymbol{\Sigma}$, we have
\begin{equation}
\boldsymbol{\Sigma}\mathbf{v}=\lambda\mathbf{v}\label{eq:rc.1}
\end{equation}
Since $\boldsymbol{\Sigma}\in\mathbb{R}^{D\times D}$, we have $\boldsymbol{\Sigma}=\overline{\boldsymbol{\Sigma}}$. Conjugate both sides of the equation above we have
\begin{equation}
\boldsymbol{\Sigma}\overline{\mathbf{v}}=\overline{\lambda}\overline{\mathbf{v}},\label{eq:rc.2}
\end{equation}
Since $\boldsymbol{\Sigma}$ is symmetric, we have $\boldsymbol{\Sigma}=\boldsymbol{\Sigma}^\text{T}$. Taking the transpose of both sides of \eqref{eq:rc.2} gives us
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}=\overline{\lambda}\overline{\mathbf{v}}^\text{T}\label{eq:rc.3}
\end{equation}
Continuing by taking dot product of both sides of \eqref{eq:rc.3} with $\mathbf{v}$ lets us obtain
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}\mathbf{v}=\overline{\lambda}\overline{\mathbf{v}}^\text{T}\mathbf{v}\label{eq:rc.4}
\end{equation}
On the other hand, take dot product of $\overline{\mathbf{v}}^\text{T}$ with both sides of \eqref{eq:rc.1}, we have
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}\mathbf{v}=\lambda\overline{\mathbf{v}}^\text{T}\mathbf{v}
\end{equation}
which by \eqref{eq:rc.4} implies that
\begin{equation}
\overline{\lambda}\overline{\mathbf{v}}^\text{T}\mathbf{v}=\lambda\overline{\mathbf{v}}^\text{T}\mathbf{v},
\end{equation}
or
\begin{equation}
(\lambda-\overline{\lambda})\overline{\mathbf{v}}^\text{T}\mathbf{v}=0\label{eq:rc.5}
\end{equation}
Moreover, we have that
\begin{equation}
\overline{\mathbf{v}}^\text{T}\mathbf{v}=\sum_{k=1}^{D}(a_k-i b_k)(a_k+i b_k)=\sum_{k=1}^{D}a^2+b^2>0
\end{equation}
where we have denoted the complex eigenvector $\mathbf{v}\neq\mathbf{0}$ as
\begin{equation}
\mathbf{v}=(a_1+i b_1,\ldots,a_D+i b_D)^\text{T},
\end{equation}
which implies that its complex conjugate $\overline{\mathbf{v}}$ can be written by
\begin{equation}
\overline{\mathbf{v}}=(a_1-i b_1,\ldots,a_D-i b_D)^\text{T}
\end{equation}
Therefore, by \eqref{eq:rc.5}, we can claim that
\begin{equation}
\lambda=\overline{\lambda}
\end{equation}
or in other words, the eigenvalue $\lambda$ of $\boldsymbol{\Sigma}$ is real.

### Projection onto eigenvectors
{: proj-ev-cov}
First, we have that eigenvectors $\mathbf{v}\_i$ and $\mathbf{v}\_j$ corresponding to different eigenvalues $\lambda_i$ and $\lambda_j$ of $\boldsymbol{\Sigma}$ are perpendicular, because
\begin{align}
\lambda_i\mathbf{v}\_i^\text{T}\mathbf{v}\_j&=\mathbf{v}\_i^\text{T}\boldsymbol{\Sigma}^\text{T}\mathbf{v}\_j \\\\ &=\mathbf{v}\_i^\text{T}\boldsymbol{\Sigma}\mathbf{v}\_j=\mathbf{v}\_i^\text{T}\lambda_j\mathbf{v}\_j,
\end{align}
which implies that
\begin{equation}
(\lambda_i-\lambda_j)\mathbf{v}\_i^\text{T}\mathbf{v}\_j=0
\end{equation}
Therefore, $\mathbf{v}\_i^\text{T}\mathbf{v}\_j=0$ since $\lambda_i\neq\lambda_j$.

Hence, for any unit eigenvectors $\mathbf{q}\_i,\mathbf{q}\_j$ of $\boldsymbol{\Sigma}$, we have
\begin{equation}
\mathbf{q}\_i^\text{T}\mathbf{q}\_j\begin{cases}1,&\hspace{0.5cm}\text{if }i=j \\\\ 0,&\hspace{0.5cm}\text{if }i\neq j\end{cases}
\end{equation}
This allows us to write $\boldsymbol{\Sigma}$ as
\begin{equation}
\boldsymbol{\Sigma}=\mathbf{Q}^\text{T}\boldsymbol{\Lambda}\mathbf{Q},
\end{equation}
where $\mathbf{Q}$ is the orthonormal matrix whose $i$-th row is $\mathbf{q}\_i^\text{T}$ and $\boldsymbol{\Lambda}$ is the diagonal matrix whose $\\{i,i\\}$ element is $\lambda_i$, as
\begin{equation}
\mathbf{Q}=\left[\begin{matrix}-\hspace{0.15cm}\mathbf{q}\_1^\text{T}\hspace{0.15cm}- \\\\ \vdots \\\\ -\hspace{0.15cm}\mathbf{q}\_D^\text{T}\hspace{0.15cm}-\end{matrix}\right],\hspace{2cm}\boldsymbol{\Lambda}=\left[\begin{matrix}\lambda_1&& \\\\ &\ddots& \\\\ &&\lambda_D\end{matrix}\right]
\end{equation}
Therefore, we can also write $\boldsymbol{\Sigma}$ as
\begin{equation}
\boldsymbol{\Sigma}=\sum_{i=1}^{D}\lambda_i\mathbf{q}\_i\mathbf{q}\_i^\text{T}
\end{equation}
Each matrix $\mathbf{q}\_i\mathbf{q}\_i^\text{T}$ is the projection matrix onto $\mathbf{q}\_i$, then $\boldsymbol{\Sigma}$ can be express as a combination of perpendicular projection matrices.

Other than that, for any eigenvector, eigenvalue pair $(\mathbf{q_i},\lambda_i)$ of the matrix $\boldsymbol{\Sigma}$, we have
\begin{align}
\lambda_i\boldsymbol{\Sigma}^{-1}\mathbf{q}\_i=\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}\mathbf{q}\_i=\mathbf{q}\_i
\end{align}
or
\begin{equation}
\boldsymbol{\Sigma}^{-1}\mathbf{q}\_i=\frac{1}{\lambda_i}\mathbf{q}\_i,
\end{equation}
<span id='precision-eigenvalue'>which implies that each eigenvector, eigenvalue pair $(\mathbf{q_i},\lambda_i)$ of $\boldsymbol{\Sigma}$ corresponds to an eigenvector, eigenvalue pair $(\mathbf{q}\_i,1/\lambda_i)$ of $\boldsymbol{\Sigma}^{-1}$. Therefore, $\boldsymbol{\Sigma}^{-1}$ can also be written by</span>
\begin{equation}
\boldsymbol{\Sigma}^{-1}=\sum_{i=1}^{D}\frac{1}{\lambda_i}\mathbf{q}\_i\mathbf{q}\_i^\text{T}\label{eq:pec.1}
\end{equation}

## Geometrical interpretation
{: #geo-int}
Consider the probability density function of the Gaussian \eqref{eq:mvn.2}, by the result \eqref{eq:pec.1}, we have that the functional dependence of the Gaussian on $\mathbf{x}$ is through the quadratic form
\begin{align}
\Delta^2&=(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \\\\ &=\sum_{i=1}^{D}\frac{y_i^2}{\lambda_i},
\end{align}
where we have defined
\begin{equation}
y_i=\mathbf{q}\_i^\text{T}(\mathbf{x}-\boldsymbol{\mu})
\end{equation}
Let $\mathbf{y}=(y_1,\ldots,y_D)^\text{T}$ be the vector comprising $y_i$'s together, then we have
\begin{equation}
\mathbf{y}=\mathbf{Q}(\mathbf{x}-\boldsymbol{\mu})
\end{equation}
Consider the form of the Gaussian distribution in the new coordinate system defined by $y_i$. When changing variable from $\mathbf{x}$ to $\mathbf{y}$, firstly we define the **Jacobian matrix** $\mathbf{J}$, whose elements are given by
\begin{equation}
\mathbf{J}\_{ij}=\frac{\partial x_i}{\partial y_j}=\mathbf{Q}\_{ji},
\end{equation}
which implies that
\begin{equation}
\mathbf{J}=\mathbf{Q}^\text{T}
\end{equation}
Thus, $\vert\mathbf{J}\vert=\vert\mathbf{Q}^\text{T}\vert=1$ since
\begin{equation}
1=\vert\mathbf{I}\vert=\vert\mathbf{Q}^\text{T}\mathbf{Q}\vert=\vert\mathbf{Q}^\text{T}\vert\vert\mathbf{Q}\vert=\vert\mathbf{Q}^\text{T}\vert
\end{equation}
Additionally, by \eqref{eq:pec.1}, we also have
\begin{equation}
\vert\boldsymbol{\Sigma}\vert^{1/2}=\left\vert\mathbf{Q}^\text{T}\boldsymbol{\Lambda}\mathbf{Q}\right\vert^{1/2}=\left(\vert\mathbf{Q}^\text{T}\vert\vert\boldsymbol{\Lambda}\vert\vert\mathbf{Q}\vert\right)^{1/2}=\prod_{i=1}^{D}\lambda_i^{1/2}
\end{equation}
Therefore, in the $y_j$ coordinate system, the Gaussian distribution takes the form
\begin{equation}
p(\mathbf{y})=\mathbf{x}\vert\mathbf{J}\vert=\prod_{j=1}^{D}\frac{1}{(2\pi\lambda_j)^{1/2}}\exp\left(-\frac{y_j^2}{2\lambda_j}\right),
\end{equation}
which is the product of $D$ independent univariate Gaussian distributions.

## Conditional Gaussian distribution
{: #cond-gauss-dist}
Let $\mathbf{x}$ be a $D$-dimensional random vector such that $\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$, and that we partition $\mathbf{x}$ into two disjoint subsets $\mathbf{x}\_a$ and $\mathbf{x}\_b$ with $\mathbf{x}\_a$ is an $M$-dimensional vector and $\mathbf{x}\_b$ is a $(D-M)$-dimensional vector.
\begin{equation}
\mathbf{x}=\left[\begin{matrix}\mathbf{x}\_a \\\\ \mathbf{x}\_b\end{matrix}\right]
\end{equation}
Along with them, we also define their corresponding means, as a partition of $\boldsymbol{\mu}$
\begin{equation}
\boldsymbol{\mu}=\left[\begin{matrix}\boldsymbol{\mu}\_a \\\\ \boldsymbol{\mu}\_b\end{matrix}\right]
\end{equation}
and their corresponding covariance matrices
\begin{equation}
\boldsymbol{\Sigma}=\left[\begin{matrix}\boldsymbol{\Sigma}\_{aa}&\boldsymbol{\Sigma}\_{ab} \\\\ \boldsymbol{\Sigma}\_{b a}&\boldsymbol{\Sigma}\_{bb}\end{matrix}\right],
\end{equation}
which implies that $\boldsymbol{\Sigma}\_{ab}=\boldsymbol{\Sigma}\_{b a}^\text{T}$.

Analogously, we also define the partitioned form of the precision matrix $\boldsymbol{\Sigma}^{-1}$
\begin{equation}
\boldsymbol{\Lambda}\doteq\boldsymbol{\Sigma}^{-1}=\left[\begin{matrix}\boldsymbol{\Lambda}\_{aa}&\boldsymbol{\Lambda}\_{ab} \\\\ \boldsymbol{\Lambda}\_{ba}&\boldsymbol{\Lambda}\_{bb}\end{matrix}\right],
\end{equation}
Thus, we also have that $\boldsymbol{\Lambda}\_{ab}=\boldsymbol{\Lambda}\_{ba}^\text{T}$ since $\boldsymbol{\Sigma}^{-1}$ or in other words, $\boldsymbol{\Lambda}$ is symmetric due to the symmetry of $\boldsymbol{\Sigma}$.
With these partitions, we can rewrite the functional dependence of the Gaussian \eqref{eq:mvn.2} on $\mathbf{x}$ as
\begin{align}
\hspace{-1cm}-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})&=-\frac{1}{2}(\mathbf{x}\_a-\boldsymbol{\mu}\_a)^\text{T}\boldsymbol{\Lambda}\_{aa}(\mathbf{x}\_a-\boldsymbol{\mu}\_a)-\frac{1}{2}(\mathbf{x}\_a-\boldsymbol{\mu}\_a)^\text{T}\boldsymbol{\Lambda}\_{ab}(\mathbf{x}\_b-\boldsymbol{\mu}\_b) \\\\ &\hspace{0.5cm}-\frac{1}{2}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)^\text{T}\boldsymbol{\Lambda}\_{ba}(\mathbf{x}\_a-\boldsymbol{\mu}\_a)-\frac{1}{2}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)^\text{T}\boldsymbol{\Lambda}\_{bb}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)\label{eq:cgd.1}
\end{align}
Consider the conditional probability $p(\mathbf{x}\_a\vert\mathbf{x}\_b)$, which is the distribution of $\mathbf{x}\_a$ given $\mathbf{x}\_b$. Viewing $\mathbf{x}\_b$ as a constant, \eqref{eq:cgd.1} will be the functional dependence of the conditional probability $p(\mathbf{x}\_a\vert\mathbf{x}\_b)$ on $\mathbf{x}\_a$, which can be continued to derive as 
\begin{align}
&-\frac{1}{2}\mathbf{x}\_a^\text{T}\boldsymbol{\Lambda}\_{aa}\mathbf{x}\_a+\frac{1}{2}\mathbf{x}\_a^\text{T}\big(\boldsymbol{\Lambda}\_{aa}\boldsymbol{\mu}\_a+\boldsymbol{\Lambda}\_{aa}^\text{T}\boldsymbol{\mu}\_a-\boldsymbol{\Lambda}\_{ab}\mathbf{x}\_b+\boldsymbol{\Lambda}\_{ab}\boldsymbol{\mu}\_b-\boldsymbol{\Lambda}\_{ba}^\text{T}\mathbf{x}\_b+\boldsymbol{\Lambda}\_{ba}\boldsymbol{\mu}\_b\big)+c \\\\ &\hspace{3cm}=-\frac{1}{2}\mathbf{x}\_a^\text{T}\boldsymbol{\Lambda}\_{aa}\mathbf{x}\_a+\mathbf{x}\_a^\text{T}\big(\boldsymbol{\Lambda}\_{aa}\boldsymbol{\mu}\_a-\boldsymbol{\Lambda}\_{ab}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)\big)+c,\label{eq:cgd.2}
\end{align}
where $c$ is a constant, and we have used the $\boldsymbol{\Lambda}\_{aa}=\boldsymbol{\Lambda}\_{aa}^\text{T}$ and $\boldsymbol{\Lambda}\_{ab}=\boldsymbol{\Lambda}\_{ba}^\text{T}$.

Moreover, we have that the variation part which depends on $\mathbf{x}$ for any Gaussian $\mathbf{X}\sim\mathcal{N}(\mathbf{x}\vert\boldsymbol{\mu},\boldsymbol{\Sigma})$ can be written as a quadratic function of $\mathbf{x}$
\begin{equation}
-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})=-\frac{1}{2}\mathbf{x}^\text{T}\boldsymbol{\Sigma}^{-1}\mathbf{x}+\mathbf{x}^\text{T}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}+c,\label{eq:cgd.3}
\end{equation}
where $c$ is a constant. With this observation, and by \eqref{eq:cgd.2} we have that the conditional distribution $p(\mathbf{x}\_a\vert\mathbf{x}\_b)$ is a Gaussian, with the corresponding covariance matrix, denoted as $\boldsymbol{\Sigma}\_{a\vert b}$, given by
\begin{equation}
\boldsymbol{\Sigma}\_{a\vert b}=\boldsymbol{\Lambda}\_{aa}^{-1},\label{eq:cgd.4}
\end{equation}
and with the corresponding mean vector, denoted as $\boldsymbol{\mu}\_{a\vert b}$, given by
\begin{align}
\boldsymbol{\mu}\_{a\vert b}&=\boldsymbol{\Sigma}\_{a\vert b}\big(\boldsymbol{\Lambda}\_{aa}\boldsymbol{\mu}\_a-\boldsymbol{\Lambda}\_{ab}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)\big) \\\\ &=\boldsymbol{\mu}\_a-\boldsymbol{\Lambda}\_{aa}^{-1}\boldsymbol{\Lambda}\_{ab}(\mathbf{x}\_b-\boldsymbol{\mu}\_b)\label{eq:cgd.5}
\end{align}
To express the mean $\boldsymbol{\mu}\_{a\vert b}$ and the covariance matrix $\boldsymbol{\Sigma}\_{a\vert b}$ of $p(\mathbf{x}\_a\vert\mathbf{x}\_b)$ in terms of partition of the covariance matrix $\boldsymbol{\Sigma}$ instead of the precision matrix $\boldsymbol{\Lambda}$'s, we will be using the identity for the inverse of a partitioned matrix
\begin{align}
\left[\begin{matrix}\mathbf{A}&\mathbf{B} \\\\ \mathbf{C}&\mathbf{D}\end{matrix}\right]^{-1}=\left[\begin{matrix}\mathbf{M}&-\mathbf{M}\mathbf{B}\mathbf{D}^{-1} \\\\ -\mathbf{D}^{-1}\mathbf{C}\mathbf{M}&\mathbf{D}^{-1}+\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}\mathbf{D}^{-1}\end{matrix}\right],\label{eq:cgd.6}
\end{align}
where we have defined
\begin{equation}
\mathbf{M}\doteq(\mathbf{A}-\mathbf{B}\mathbf{D}^{-1}\mathbf{C})^{-1},
\end{equation}
whose inverse $\mathbf{M}^{-1}$ is called the **Schur complement** of the matrix $\left[\begin{matrix}\mathbf{A}&\mathbf{B} \\\\ \mathbf{C}&\mathbf{D}\end{matrix}\right]^{-1}$. This identity can be proved by multiplying both sides of \eqref{eq:cgd.6} with $\left[\begin{matrix}\mathbf{A}&\mathbf{B} \\\\ \mathbf{C}&\mathbf{D}\end{matrix}\right]$ to give
\begin{align}
\mathbf{I}&=\left[\begin{matrix}\mathbf{M}&-\mathbf{M}\mathbf{B}\mathbf{D}^{-1} \\\\ -\mathbf{D}^{-1}\mathbf{C}\mathbf{M}&\mathbf{D}^{-1}+\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}\mathbf{D}^{-1}\end{matrix}\right]\left[\begin{matrix}\mathbf{A}&\mathbf{B} \\\\ \mathbf{C}&\mathbf{D}\end{matrix}\right] \\\\ &=\left[\begin{matrix}\mathbf{M}(\mathbf{A}-\mathbf{B}\mathbf{D}^{-1}\mathbf{C})&\mathbf{M}\mathbf{B}-\mathbf{M}\mathbf{B} \\\\ -\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{A}+\mathbf{D}^{-1}\mathbf{C}+\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}\mathbf{D}^{-1}\mathbf{C}&-\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}+\mathbf{I}+\mathbf{D}^{-1}\mathbf{C}\mathbf{M}\mathbf{B}\end{matrix}\right] \\\\ &=\left[\begin{matrix}\mathbf{I}&\mathbf{0} \\\\ \mathbf{D}^{-1}\mathbf{C}\big(\mathbf{I}-\mathbf{M}(\mathbf{A}-\mathbf{B}\mathbf{D}^{-1}\mathbf{C})\big)&\mathbf{I}\end{matrix}\right] \\\\ &=\left[\begin{matrix}\mathbf{I}&\mathbf{0} \\\\ \mathbf{0}&\mathbf{I}\end{matrix}\right]=\mathbf{I},
\end{align}
which claims our argument.

Applying the identity \eqref{eq:cgd.6} into the precision matrix $\boldsymbol{\Lambda}=\boldsymbol{\Sigma}^{-1}$ gives us
\begin{equation}
\hspace{-0.5cm}\left[\begin{matrix}\boldsymbol{\Lambda}\_{aa}&\boldsymbol{\Lambda}\_{ab} \\\\ \boldsymbol{\Lambda}\_{ba}&\boldsymbol{\Lambda}\_{bb}\end{matrix}\right]=\left[\begin{matrix}\boldsymbol{\Sigma}\_{aa}&\boldsymbol{\Sigma}\_{ab} \\\\ \boldsymbol{\Sigma}\_{b a}&\boldsymbol{\Sigma}\_{bb}\end{matrix}\right]^{-1}=\left[\begin{matrix}\mathbf{M}\_\boldsymbol{\Sigma}&-\mathbf{M}\_\boldsymbol{\Sigma}\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1} \\\\ -\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}\mathbf{M}\_\boldsymbol{\Sigma}&\boldsymbol{\Sigma}\_{bb}^{-1}+\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}\mathbf{M}\_\boldsymbol{\Sigma}\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\end{matrix}\right],
\end{equation}
where the Schur complement of $\mathbf{\Sigma}^{-1}$ is given by
\begin{equation}
\mathbf{M}\_\boldsymbol{\Sigma}=\big(\boldsymbol{\Sigma}\_{aa}-\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}\big)^{-1}
\end{equation}
Hence, we obtain
\begin{align}
\boldsymbol{\Lambda}\_{aa}&=\mathbf{M}\_\boldsymbol{\Sigma}=\big(\boldsymbol{\Sigma}\_{aa}-\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}\big)^{-1}, \\\\ \boldsymbol{\Lambda}\_{ab}&=-\mathbf{M}\_\boldsymbol{\Sigma}\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}=\big(\boldsymbol{\Sigma}\_{aa}-\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}\big)^{-1}\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}
\end{align}
Substitute these results into \eqref{eq:cgd.4} and \eqref{eq:cgd.5}, we have the mean and the covariance matrix of the conditional Gaussian distribution $p(\mathbf{x}\_a\vert\mathbf{x}\_b)$ can be rewritten as
\begin{align}
\boldsymbol{\mu}\_{a\vert b}&=\boldsymbol{\mu}\_a+\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}(\mathbf{x}\_b-\boldsymbol{\mu}\_b), \\\\ \boldsymbol{\Sigma}\_{a\vert b}&=\boldsymbol{\Sigma}\_{aa}-\boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}
\end{align}
It is worth noticing that the mean $\boldsymbol{\mu}\_{a\vert b}$ given above is a linear function of $\mathbf{x}\_b$, while the covariance matrix $\boldsymbol{\Sigma}\_{a\vert b}$ is independent of $\mathbf{x}\_b$. This is an example of a **linear-Gaussian model**.

## Marginal Gaussian distribution
{: #marg-gauss-dist}

## Bayes' theorem for Gaussian variables
{: #bayes-theorem-gauss}
In this section, we will apply the Bayes' theorem to find the marginal distribution of $p(\mathbf{y})$ and conditional distribution $p(\mathbf{x}\vert\mathbf{y})$ with supposing that we are given a Gaussian distribution $p(\mathbf{x})$ and a conditional Gaussian distribution $p(\mathbf{y}\vert\mathbf{x})$ in which $p(\mathbf{y}\vert\mathbf{x})$ has a mean that is a linear function of $\mathbf{x}$, and a covariance matrix which is independent of $\mathbf{x}$, as
\begin{align}
p(\mathbf{x})&=\mathcal{N}(\mathbf{x}\vert\boldsymbol{\mu},\boldsymbol{\Lambda}^{-1}), \\\\ p(\mathbf{y}\vert\mathbf{x})&=\mathcal{N}(\mathbf{y}\vert\mathbf{A}\mathbf{x}+\mathbf{b},\mathbf{L}^{-1}),
\end{align}
where $\mathbf{A},\mathbf{b}$ are two parameters controlling the means, and $\boldsymbol{\Lambda},\boldsymbol{L}$ are precision matrices.

In order to find the marginal and conditional distribution, first we will be looking for the joint distribution $p(\mathbf{x},\mathbf{y})$ by considering the augmented vector
\begin{equation}
\mathbf{z}=\left[\begin{matrix}\mathbf{x} \\\\ \mathbf{y}\end{matrix}\right]
\end{equation}
Therefore, we have
\begin{equation}
p(\mathbf{z})=p(\mathbf{x},\mathbf{y})=p(\mathbf{x})p(\mathbf{y}\vert\mathbf{x})
\end{equation}
Taking the natural logarithm of both sides gives us
\begin{align}
\log p(\mathbf{z})&=\log p(\mathbf{x})+\log p(\mathbf{y}\vert\mathbf{x}) \\\\ &=\log\mathcal{N}(\mathbf{x}\vert\boldsymbol{\mu},\boldsymbol{\Lambda}^{-1})+\log\mathcal{N}(\mathbf{y}\vert\mathbf{A}\mathbf{x}+\mathbf{b},\mathbf{L}^{-1}) \\\\ &=-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Lambda}(\mathbf{x}-\boldsymbol{\mu})-\frac{1}{2}(\mathbf{y}-\mathbf{A}\mathbf{x}-\mathbf{b})^\text{T}\mathbf{L}(\mathbf{y}-\mathbf{A}\mathbf{x}-\mathbf{b})+c\label{eq:btg.1}
\end{align}
where $c$ is a constant in terms of $\mathbf{x}$ and $\mathbf{y}$, i.e., $c$ is independent of $\mathbf{x},\mathbf{y}$.

It is easily to notice that \eqref{eq:btg.1} is a quadratic function of the components of $\mathbf{z}$, which implies that $p(\mathbf{z})$ is a Gaussian. By \eqref{eq:cgd.3}, in order to find the covariance matrix of $\mathbf{z}$, we consider the quadratic terms in \eqref{eq:btg.1}, which are given by
\begin{align}
&-\frac{1}{2}\mathbf{x}^\text{T}\boldsymbol{\Lambda}\mathbf{x}-\frac{1}{2}(\mathbf{y}-\mathbf{A}\mathbf{x})^\text{T}\mathbf{L}(\mathbf{y}-\mathbf{A}\mathbf{x}) \\\\ &=-\frac{1}{2}\Big[\mathbf{x}^\text{T}\big(\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A}\big)\mathbf{x}+\mathbf{y}^\text{T}\mathbf{L}\mathbf{y}-\mathbf{y}^\text{T}\mathbf{L}\mathbf{A}\mathbf{x}-\mathbf{x}^\text{T}\mathbf{A}^\text{T}\mathbf{L}\mathbf{y}\Big] \\\\ &=-\frac{1}{2}\left[\begin{matrix}\mathbf{x} \\\\ \mathbf{y}\end{matrix}\right]^\text{T}\left[\begin{matrix}\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A}&-\mathbf{A}^\text{T}\mathbf{L} \\\\ -\mathbf{L}\mathbf{A}&\mathbf{L}\end{matrix}\right]\left[\begin{matrix}\mathbf{x} \\\\ \mathbf{y}\end{matrix}\right] \\\\ &=-\frac{1}{2}\mathbf{z}^\text{T}\mathbf{R}\mathbf{z},
\end{align}
which implies that the precision matrix of $\mathbf{z}$ is $\mathbf{R}$, defined as
\begin{equation}
\mathbf{R}=\left[\begin{matrix}\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A}&-\mathbf{A}^\text{T}\mathbf{L} \\\\ -\mathbf{L}\mathbf{A}&\mathbf{L}\end{matrix}\right]
\end{equation}
Thus, using the identity \eqref{eq:cgd.6}, we obtain the covariance matrix of the joint distribution
\begin{equation}
\boldsymbol{\Sigma}\_\mathbf{z}=\mathbf{R}^{-1}=\left[\begin{matrix}\boldsymbol{\Lambda}^{-1}&\boldsymbol{\Lambda}^{-1}\mathbf{A}^\text{T} \\\\ \mathbf{A}\boldsymbol{\Lambda}^{-1}&\mathbf{L}^{-1}+\mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^\text{T}\end{matrix}\right]
\end{equation}
Analogously, by \eqref{eq:cgd.3}, we can find the mean of the joint distribution by considering the linear terms of \eqref{eq:btg.1}, which are
\begin{align}
\hspace{-0.7cm}\frac{1}{2}\Big[\mathbf{x}^\text{T}\boldsymbol{\Lambda}\boldsymbol{\mu}+\boldsymbol{\mu}^\text{T}\boldsymbol{\Lambda}\mathbf{x}+(\mathbf{y}-\mathbf{A}\mathbf{x})^\text{T}\mathbf{L}\mathbf{b}+\mathbf{b}^\text{T}\mathbf{L}(\mathbf{y}-\mathbf{A}\mathbf{x}) \Big]&=\mathbf{x}^\text{T}\boldsymbol{\Lambda}\boldsymbol{\mu}-\mathbf{x}^\text{T}\mathbf{A}^\text{T}\mathbf{L}\mathbf{b}+\mathbf{y}^\text{T}\mathbf{L}\mathbf{b} \\\\ &=\left[\begin{matrix}\mathbf{x} \\\\ \mathbf{y}\end{matrix}\right]^\text{T}\left[\begin{matrix}\boldsymbol{\Lambda}\boldsymbol{\mu}-\mathbf{A}^\text{T}\mathbf{L}\mathbf{b} \\\\ \mathbf{L}\mathbf{b}\end{matrix}\right]
\end{align}
Thus, by \eqref{eq:cgd.3}, we have that the mean of the joint distribution is then given by
\begin{equation}
\boldsymbol{\mu}\_\mathbf{z}=\boldsymbol{\Sigma}\_\mathbf{z}\left[\begin{matrix}\boldsymbol{\Lambda}\boldsymbol{\mu}-\mathbf{A}^\text{T}\mathbf{L}\mathbf{b} \\\\ \mathbf{L}\mathbf{b}\end{matrix}\right]=\left[\begin{matrix}\boldsymbol{\mu} \\\\ \mathbf{A}\boldsymbol{\mu}+\mathbf{b}\end{matrix}\right]
\end{equation}
Given the mean $\boldsymbol{\mu}\_\mathbf{z}$ and the covariance matrix $\boldsymbol{\Sigma}\_\mathbf{z}$ of the joint distribution of $\mathbf{x},\mathbf{y}$, by \eqref{22} and \eqref{23}, we then can obtain the mean of the covariance matrix of the marginal distribution $p(\mathbf{y})$, which are
\begin{align}
\boldsymbol{\mu}\_\mathbf{y}&=\mathbf{A}\boldsymbol{\mu}+\mathbf{b}, \\\\ \boldsymbol{\Sigma}\_\mathbf{y}&=\mathbf{L}^{-1}+\mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^\text{T},
\end{align}
and also, by \eqref{eq:cgd.4} and \eqref{eq:cgd.5}, we can easily get mean and covariance matrix of the conditional distribution $p(\mathbf{x}\vert\mathbf{y})$, which are given by
\begin{align}
\boldsymbol{\mu}\_{\mathbf{x}\vert\mathbf{y}}&=(\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A})^{-1}\big(\mathbf{A}^\text{T}\mathbf{L}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Lambda}\boldsymbol{\mu}\big) \\\\ \boldsymbol{\Sigma}\_{\mathbf{x}\vert\mathbf{y}}&=(\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A})^{-1}
\end{align}
In Bayesian approach, we can consider $p(\mathbf{x})$ as a prior distribution over $\mathbf{x}$, and if $\mathbf{y}$ is observed, the conditional distribution $p(\mathbf{x}\vert\mathbf{y})$ will represents the corresponding posterior distribution over $\mathbf{x}$.

<span id='marg-cond-gaussian'>**Remark**.  
Given a marginal Gaussian distribution for $\mathbf{x}$ and a conditional Gaussian distribution for $\mathbf{y}$ given $\mathbf{x}$ in the form
\begin{align}
p(\mathbf{x})&=\mathcal{N}(\mathbf{x}\vert\boldsymbol{\mu},\boldsymbol{\Lambda}^{-1}), \\\\ p(\mathbf{y}\vert\mathbf{x})&=\mathcal{N}(\mathbf{y}\vert\mathbf{A}\mathbf{x}+\mathbf{b},\mathbf{L}^{-1}),
\end{align}
the marginal distribution of $\mathbf{y}$ and the conditional distribution of $\mathbf{x}$ given $\mathbf{y}$ are then given by
\begin{align}
p(\mathbf{y})&=\mathcal{N}(\mathbf{y}\vert\mathbf{A}\boldsymbol{\mu}+\mathbf{b},\mathbf{L}^{-1}+\mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^\text{T}), \\\\ p(\mathbf{x}\vert\mathbf{y})&=\mathcal{N}(\mathbf{x}\vert\boldsymbol{\Sigma}(\mathbf{A}^\text{T}\mathbf{L}(\mathbf{y}-\mathbf{b})+\boldsymbol{\Lambda}\boldsymbol{\mu}),\boldsymbol{\Sigma})
\end{align}
where
\begin{equation}
\boldsymbol{\Sigma}=(\boldsymbol{\Lambda}+\mathbf{A}^\text{T}\mathbf{L}\mathbf{A})^{-1}
\end{equation}</span>

## References
{: #references}
[1] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573).  

[2] Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732). Springer New York, NY, 2006. 

[3] Gilbert Strang. [Introduction to Linear Algebra, 5th edition](http://math.mit.edu/~gs/linearalgebra/), 2016.

## Footnotes
{: #footnotes}
[^1]: The definition of covariance matrix $\boldsymbol{\Sigma}$ can be rewritten as
	\begin{equation}
	\boldsymbol{\Sigma}=\Cov(\mathbf{X},\mathbf{X})=\Var(\mathbf{X})
	\end{equation}
	Let $\mathbf{z}\in\mathbb{R}^D$, we have
	\begin{equation}
	\Var(\mathbf{z}^\text{T}\mathbf{X})=\mathbf{z}^\text{T}\Var(\mathbf{X})\mathbf{z}=\mathbf{z}^\text{T}\boldsymbol{\Sigma}\mathbf{z}
	\end{equation}
	And since $\Var(\mathbf{z}^\text{T}\mathbf{X})\geq0$, we also have that $\mathbf{z}^\text{T}\mathbf{\Sigma}\mathbf{z}\geq0$, which proves that $\boldsymbol{\Sigma}$ is a positive semi-definite matrix.