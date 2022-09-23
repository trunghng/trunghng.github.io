---
layout: post
title:  "Gaussian Distribution"
date:   2021-11-22 14:46:00 +0700
categories: mathematics probability-statistics
tags: mathematics probability-statistics normal-distribution
description: Gaussian Distribution
comments: true
---
> A note on Gaussian distribution.
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
- [References](#references)
- [Footnotes](#footnotes)  

$\newcommand{\Var}{\mathrm{Var}}$
$\newcommand{\Cov}{\mathrm{Cov}}$
## Gaussian (Normal) Distribution
{: #gauss-dist}
A random variable $X$ is said to be **Gaussian** or to have the **Normal distribution** with mean $\mu$ and variance $\sigma^2$ if its probability density function (PDF) is
\begin{equation}
f_X(x)=\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)\tag{1}\label{1}
\end{equation}
which we denote as $X\sim\mathcal{N}(\mu,\sigma)$.

### Standard Normal
{: #std-normal}
When $X$ is normally distributed with mean $\mu=0$ and variance $\sigma^2=1$, we call its distribution **Standard Normal**.
\begin{equation}
X\sim\mathcal{N}(0,1)\tag{2}\label{2}
\end{equation}
In this case, $X$ has special notations to denote its PDF and CDF, which are
\begin{equation}
\varphi(x)=\dfrac{1}{\sqrt{2\pi}}e^{-z^2/2},\tag{3}\label{3}
\end{equation}
and
\begin{equation}
\Phi(x)=\int_{-\infty}^{x}\varphi(t)\,dt=\int_{-\infty}^{x}\dfrac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt\tag{4}\label{4}
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
\mathbf{X}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})\tag{5}\label{5}
\end{equation}
where
\begin{equation}
\boldsymbol{\mu}=\mathbb{E}\mathbf{X}=\mathbb{E}\left(\mu_1,\ldots,\mu_k\right)^\text{T}=\left(\mathbb{E}X_1,\ldots,\mathbb{E}X_k\right)^\text{T}
\end{equation}
is the $D$-dimensional mean vector, and covariance matrix $\mathbf{\Sigma}\in\mathbb{R}^{D\times D}$ with
\begin{equation}
\boldsymbol{\Sigma}\_{ij}=\mathbb{E}\left(X_i-\mu_i\right)\left(X_j-\mu_j\right)=\Cov(X_i,X_j)\tag{6}\label{6}
\end{equation}
We also have that $\boldsymbol{\Sigma}\geq 0$ (positive semi-definite matrix)[^1].

Thus, the PDF of an MVN is defined as
\begin{equation}
f_\mathbf{X}(x_1,\ldots,x_D)=\dfrac{1}{(2\pi)^{D/2}\vert\mathbf{\Sigma}\vert^{1/2}}\exp\left[-\dfrac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]\tag{7}\label{7}
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
With the definition \eqref{6} of the covariance matrix $\boldsymbol{\Sigma}$, we can easily see that it is symmetric. However, notice that in the illustration of BVN, we gave the distribution a non-symmetric covariance matrix. The reason why we could do that is without loss of generality, we can assume that $\boldsymbol{\Sigma}$ is symmetric.

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

Thus, when computing th{e density, the symmetric part of $\boldsymbol{\Sigma}^{-1}$ is the only one matters. Or in other words, without loss of generality, we can assume that $\boldsymbol{\Sigma}^{-1}$ is symmetric, which means that $\boldsymbol{\Sigma}$ is also symmetric.

### Real eigenvalues
{: #re-cov}
Consider an eigenvector, eigenvalue pair $(\mathbf{v},\lambda)$ of covariance matrix $\boldsymbol{\Sigma}$, we have
\begin{equation}
\boldsymbol{\Sigma}\mathbf{v}=\lambda\mathbf{v}\tag{8}\label{8}
\end{equation}
Since $\boldsymbol{\Sigma}\in\mathbb{R}^{D\times D}$, we have $\boldsymbol{\Sigma}=\overline{\boldsymbol{\Sigma}}$. Conjugate both sides of the equation above we have
\begin{equation}
\boldsymbol{\Sigma}\overline{\mathbf{v}}=\overline{\lambda}\overline{\mathbf{v}},\tag{9}\label{9}
\end{equation}
Since $\boldsymbol{\Sigma}$ is symmetric, we have $\boldsymbol{\Sigma}=\boldsymbol{\Sigma}^\text{T}$. Taking the transpose of both sides of \eqref{9} gives us
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}=\overline{\lambda}\overline{\mathbf{v}}^\text{T}\tag{10}\label{10}
\end{equation}
Continuing by taking dot product of both sides of \eqref{10} with $\mathbf{v}$ lets us obtain
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}\mathbf{v}=\overline{\lambda}\overline{\mathbf{v}}^\text{T}\mathbf{v}\tag{11}\label{11}
\end{equation}
On the other hand, take dot product of $\overline{\mathbf{v}}^\text{T}$ with both sides of \eqref{8}, we have
\begin{equation}
\overline{\mathbf{v}}^\text{T}\boldsymbol{\Sigma}\mathbf{v}=\lambda\overline{\mathbf{v}}^\text{T}\mathbf{v}
\end{equation}
which by \eqref{11} implies that
\begin{equation}
\overline{\lambda}\overline{\mathbf{v}}^\text{T}\mathbf{v}=\lambda\overline{\mathbf{v}}^\text{T}\mathbf{v},
\end{equation}
or
\begin{equation}
(\lambda-\overline{\lambda})\overline{\mathbf{v}}^\text{T}\mathbf{v}=0\tag{12}\label{12}
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
Therefore, by \eqref{12}, we can claim that
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
\boldsymbol{\Sigma}=\mathbf{Q}^\text{T}\boldsymbol{\Lambda}\mathbf{Q},\tag{13}\label{13}
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
which implies that each eigenvector, eigenvalue pair $(\mathbf{q_i},\lambda_i)$ of $\boldsymbol{\Sigma}$ corresponds to an eigenvector, eigenvalue pair $(\mathbf{q}\_i,1/\lambda_i)$ of $\boldsymbol{\Sigma}^{-1}$. Therefore, $\boldsymbol{\Sigma}^{-1}$ can also be written by
\begin{equation}
\boldsymbol{\Sigma}^{-1}=\sum_{i=1}^{D}\frac{1}{\lambda_i}\mathbf{q}\_i\mathbf{q}\_i^\text{T}\tag{14}\label{14}
\end{equation}

## Geometrical interpretation
{: #geo-int}
Consider the probability density function of the Gaussian \eqref{7}, by the result \eqref{14}, we have that the functional dependence of the Gaussian on $\mathbf{x}$ is through the quadratic form
\begin{equation}
\Delta^2=(\mathbf{x}-\boldsymbol{\mu})^\text{T}\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})=\sum_{i=1}^{D}\frac{y_i^2}{\lambda_i},
\end{equation}
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
Additionally, by \eqref{14}, we also have
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


## References
{: #references}
[1] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573).

[2] Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732). Springer New York, NY. 

[3] Gilbert Strang. [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/).

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