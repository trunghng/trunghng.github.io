---
layout: post
title:  "Bayesian Optimization"
date:   2021-11-22 14:46:00 +0700
categories: artificial-intelligent machine-learning
tags: artificial-intelligent machine-learning gaussian-process optimization-control probability-statistics
description: bayesian optimization
comments: true
---
> Dont know yet

<!-- excerpt-end -->
- [Mathematical Basics](#basics)
	- [Gaussian (Normal) Distribution](#gauss-dist)
		- [Standard Normal](#std-norm)
	- [Multivariate Normal Distribution](#mvn)
		- [Bivariate Normal](#bvn)
	- [Kernels](#kernels)
		- [RBF kernels](#rbf-kernels)
		- [Mercer (positive definite) kernels](#mercer-kernels)
		- [Linear kernels](#lin-kernels)
		- [Matern kernels](#matern-kernels)
- [Gaussian Process](#gp)
	[Gaussian Process Regression](#gpr)
- [Bayesian Opitmization](#bayes-opt)
	- [Surrogate Model](#surrogate-model)
	- [Acquisition Functions](#acquisition-func)
	- [Optimization Algorithm](#opt-alg)
	- [Expected Improvement](#exp-imp)
	- [Implementation](#implementation)
- [References](#references)
- [Footnotes](#footnotes)  


Before diving into details, we need some necessary basic concepts.


## Background
{: #basics}


### Gaussian (Normal) Distribution
{: #gauss-dist}
A random variable $X$ is said to be **Gaussian** or to have the **Normal distribution** with mean $\mu$ and variance $\sigma^2$ if its probability density function (PDF) is
\begin{equation}
f_X(x)=\dfrac{1}{\sqrt(2\pi)\sigma}\exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)
\end{equation}
which we denote as $X\sim\mathcal{N}(\mu,\sigma)$


#### Standard Normal
{: #std-normal}
When $X$ is normally distributed with mean $\mu=0$ and variance $\sigma^2=1$, we call its distribution *Standard Normal*.
\begin{equation}
X\sim\mathcal{N}(0,1)
\end{equation}
In this case, $X$ has special notations to denote its PDF and CDF, which are
\begin{equation}
\varphi(x)=\dfrac{1}{\sqrt{2\pi}}e^{-z^2/2}
\end{equation}
\begin{equation}
\Phi(x)=\int_{-\infty}^{x}\varphi(t)\,dt=\int_{-\infty}^{x}\dfrac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt
\end{equation}
Below is some visualizations of Normal distribution.

<figure>
	<img src="/assets/images/2021-11-22/normal.png" alt="normal distribution" width="900" height="380px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: 10K normally distributed data points (5K each plot) were plotted as vertical bars on x-axis. The code can be found <span markdown="1">[here](https://github.com/trunghng/bayes-opt/blob/main/gauss-dist.py)</span></figcaption>
</figure><br/>


### Multivariate Normal Distribution
{: #mvn}
A $k$-dimensional random vector $\mathbf{X}=(X_1,\dots,X_k)^T$ is said to have a **Multivariate Normal (MVN)** distribution if every linear combination of the $X_i$ has a Normal distribution. Which means
\begin{equation}
t_1X_1+\ldots+t_kX_k
\end{equation}
is normally distributed for any choice of constants $t_1,\dots,t_k$. Distribution of $\mathbf{X}$ then can be written in the following notation
\begin{equation}
\mathbf{X}\sim\mathcal{N}(\mathbf{\mu},\mathbf{\Sigma})
\end{equation}
where
\begin{equation}
	\mathbf{\mu}=\mathbb{E}\mathbf{X}=\mathbb{E}\left(\mu_1,\ldots,\mu_k\right)^T=\left(\mathbb{E}X_1,\ldots,\mathbb{E}X_k\right)^T
\end{equation}
is the $k$-dimensional mean vector, and covariance matrix $\mathbf{\Sigma}\in\mathbb{R}^{k\times k}$ with
\begin{equation}
	\mathbf{\Sigma}\_{ij}=\mathbb{E}\left(X_i-\mu_i\right)\left(X_j-\mu_j\right)=\text{Cov}(X_i,X_j)
\end{equation}
We also have that $\mathbf{\Sigma}\geq 0$ (positive semi-definite matrix)[^1].

Thus, the PDF of an MVN is defined as
\begin{equation}
f_X(x_1,\ldots,x_k)=\dfrac{1}{(2\pi)^{k/2}\vert\mathbf{\Sigma}\vert^{1/2}}\exp\left[\dfrac{1}{2}(\mathbf{x}-\mathbf{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right]
\end{equation}
With this idea, *Standard Normal* distribution in multi-dimensional case can be defined as a Gaussian with mean $\mathbf{\mu}=0$ (here $0$ is an $k$-dimensional vector) and identity covariance matrix $\mathbf{\Sigma}=\mathbf{I}\_{k\times k}$.


#### Bivariate Normal
{: #bvn}
When the number of dimensions in $\mathbf{X}$, $k=2$, this special case of MVN is called the **Bivariate Normal (BVN)**. An example of an BVN is shown as following.  

<figure>
	<img src="/assets/images/2021-11-22/bvn.png" alt="monte carlo method" width="750" height="350px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The PDF of $\mathcal{N}\left(\left[\begin{smallmatrix}0\\0\end{smallmatrix}\right],\left[\begin{smallmatrix}1&0.5\\0.8&1\end{smallmatrix}\right]\right)$. The code can be found <span markdown="1">[here](https://github.com/trunghng/bayes-opt/blob/main/mvn.py)</span></figcaption>
</figure><br/>


### Kernels
**Kernel function** is a real-valued function of two arguments
\begin{equation}
\kappa(\textbf{x},\textbf{x}')\in\mathbb{R},
\end{equation}
for $\textbf{x},\textbf{x}'\in\mathcal{X}$, which typically is symmetric (i.e., $\kappa(\textbf{x},\textbf{x}')=\kappa(\textbf{x}',\textbf{x})$), and nonnegative ($\kappa(\textbf{x},\textbf{x}')\geq0$). These are some examples of kernel functions.  


#### RBF kernels
The **squared exponential kernel** (SE kernel) or **Gaussian kernel** is defined by
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\exp\left(-\frac{1}{2}(\textbf{x}-\textbf{x}')^T\mathbf{\Sigma}^{-1}(\textbf{x}-\textbf{x}')\right)
\end{equation}
If $\mathbf{\Sigma}$ is a diagonal matrix, this can be written as
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\exp\left(-\frac{1}{2}\sum_{j=1}^{D}\frac{1}{\sigma_j^2}(x_j-x_j')^2\right)
\end{equation}
We can interpret the $\sigma_j$ as defining the **characteristic length scale** of dimension $j$. The corresponding dimension of $\sigma_j$ is ignored if $\sigma_j=\infty$. This is known as the **ARD kernel** (Automatic Relevance Determination).  
If $\Sigma$ is spherical[^2], we get the isotropic kernel
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\exp\left(-\frac{\Vert\textbf{x}-\textbf{x}'\Vert}{2\sigma^2}\right)\tag{1}\label{1}
\end{equation}
Here $\sigma^2$ is known as the **bandwidth**. Since \eqref{1} can be seen as a function of $\Vert\textbf{x}-\textbf{x}'\Vert$, it is an example of a **radial basis function** or **RBF kernel**.


#### Mercer (positive definite) kernels
{: #mercer-kernels}
If the Gram matrix, defined by
\begin{equation}
\mathbf{K}=\begin{bmatrix}\kappa(\mathbf{x}\_1,\mathbf{x}\_1)&\ldots&\kappa(\mathbf{x}\_1,\mathbf{x}\_N) \\\\&\vdots&\\\\\kappa(\mathbf{x}\_N,\mathbf{x}\_1)&\ldots&\kappa(\mathbf{x}\_N,\mathbf{x}\_N)\end{bmatrix},
\end{equation}
is positive definite for any set $\left\\{x_i\right\\}\_{i=1}^N$, the kernel $\kappa$ is called a **Mercer kernel**, or **positive definite kernel**.  

The importance of Mercer kernels is the following reusult, which is known as  **Mercer's theorem**. If the Gram matrix is positive definite, we can compute an eigenvector decomposition of it as
\begin{equation}
\mathbf{K}=\mathbf{U}^T\mathbf{\Lambda}\mathbf{U},
\end{equation}
where $\mathbf{\Lambda}$ is a diagonal matrix of eigenvalues $\lambda_i>0$. Consider an element of $\mathbf{K}$
\begin{equation}
k_{ij}=\left(\mathbf{\Lambda}^{\frac{1}{2}}\mathbf{U}\_{:i}\right)^T\left(\mathbf{\Lambda}^{\frac{1}{2}}\mathbf{U}\_{:j}\right)
\end{equation}
If we let $\phi(\textbf{x}\_i)=\mathbf{\Lambda}^{\frac{1}{2}}\mathbf{U}\_{:i}$, then we can write
\begin{equation}
k_{ij}=\phi(\textbf{x}\_i)^T\phi(\textbf{x}\_j)
\end{equation}
Thus we see that the intries in the kernel matrix can be computed by performing an inner product of some feature vectors that are implicitly defined by the eigenvectors $\mathbf{U}$. In general, if the kernel is Mercer, then there exists a function $\phi$ mapping $\textbf{x}\in\mathcal{X}$ to $\mathbb{R}^D$ such that
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\phi(\textbf{x})^T\phi(\textbf{x}'),
\end{equation}
where $\phi$ depends on the eigenfunctions of $\kappa$.


#### Linear kernels
{: #lin-kernels}
Deriving the feature vector implied by a kernel is only possible if the kernel is *Mercer*. However, deriving a kernel from a feature vector is easy. We have
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\phi(\textbf{x})^T\phi(\textbf{x}')
\end{equation}
If $\phi(\textbf{x})=\textbf{x}$, we obtain the **linear kernel**
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\textbf{x}^T\textbf{x}'
\end{equation}


#### Matern kernels
{: #matern-kernels}
The **Matern kenel** has the following form
\begin{equation}
\kappa(r)=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{l}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}r}{l}\right),
\end{equation}
where $r=\Vert\textbf{x}-\textbf{x}'\Vert,\nu>0,l>0$ and $K_{\nu}$ is a modified Bessel function. As $\nu\to\infty$, this approaches the SE kernel.


## Gaussian Process
{: #gp}
Consider a model defined in terms of a linear combination of $M$ fixed basis functions given by the elements of the vecotr $\phi(x)$ so that
\begin{equation}
y(x)=w^T\phi(x)
\end{equation}





### Gaussian Process Regression
{: #gpr}


## References
[1] C. E. Rasmussen & C. K. I. Williams. [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/), MIT Press, 2006  

[2] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573)  

[3] Stanford CS229. [Machine Learning](http://cs229.stanford.edu)  

[4] Kevin P. Murphy. [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/book0.html), MIT Press, 2012  

[5] Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Setatistics/dp/0387310738)

[6] Peter I. Frazier. [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)  

[7] Martin Krasser. [Bayesian Optimization](https://krasserm.github.io/2018/03/21/bayesian-optimization/)

[8] [amoeba](https://stats.stackexchange.com/users/28666/amoeba), [What is an isotropic (spherical) covariance matrix?](https://stats.stackexchange.com/q/204599), StackExchange

## Footnotes
[^1]: 
[^2]: A covariance matrix $\mathbf{C}$ is called *isotrophic*, or *spherical* if it is proportionate to the identity matrix
	\begin{equation}
	\mathbf{C}=\lambda\mathbf{I}
	\end{equation}  
