---
layout: post
title:  "Bayesian Optimization"
date:   2021-11-22 14:46:00 +0700
categories: artificial-intelligent machine-learning
tags: artificial-intelligent machine-learning gaussian-process optimization-control probability-statistics random-stuffs
description: Bayesian optimization
comments: true
---
> This is simply a random post. Because I litterally had forgotten almost every details about these concepts until a friend of mine gave me a reason to lay my hands on these stuffs again.

<!-- excerpt-end -->
- [Preliminaries](#preliminaries)
	- [Gaussian (Normal) Distribution](#gauss-dist)
		- [Standard Normal](#std-norm)
	- [Multivariate Normal Distribution](#mvn)
		- [Bivariate Normal](#bvn)
	- [Kernels](#kernels)
		- [Kernel functions](#kernel-func)
			- [RBF kernels](#rbf-kernels)
			- [Mercer (positive definite) kernels](#mercer-kernels)
			- [Linear kernels](#lin-kernels)
			- [Matern kernels](#matern-kernels)
		- [Eigenfunction A nalysis of Kernels](#eigenfunc-kernel)
			- [Eigenfunctions](#eigenfunc)
- [Gaussian Process](#gp)
	- [Gaussian Process Regression](#gpr)
- [Bayesian Opitmization](#bayes-opt)
	- [Surrogate Model](#surrogate-model)
	- [Acquisition Functions](#acquisition-func)
	- [Optimization Algorithm](#opt-alg)
	- [Expected Improvement](#exp-imp)
	- [Implementation](#implementation)
- [References](#references)
- [Footnotes](#footnotes)  

$\newcommand{\Var}{\mathrm{Var}}$
$\newcommand{\Cov}{\mathrm{Cov}}$
## Preliminaries
{: #preliminaries}
Before diving into details, it is necessary to equip some basic concepts.


### Gaussian (Normal) Distribution
{: #gauss-dist}
A random variable $X$ is said to be **Gaussian** or to have the **Normal distribution** with mean $\mu$ and variance $\sigma^2$ if its probability density function (PDF) is
\begin{equation}
f_X(x)=\dfrac{1}{\sqrt{2\pi}\sigma}\exp\left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)
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
A $k$-dimensional random vector $\mathbf{X}=\left(X_1,\dots,X_k\right)^\intercal$ is said to have a **Multivariate Normal (MVN)** distribution if every linear combination of the $X_i$ has a Normal distribution. Which means
\begin{equation}
t_1X_1+\ldots+t_kX_k
\end{equation}
is normally distributed for any choice of constants $t_1,\dots,t_k$. Distribution of $\mathbf{X}$ then can be written in the following notation
\begin{equation}
\mathbf{X}\sim\mathcal{N}(\mathbf{\mu},\mathbf{\Sigma})
\end{equation}
where
\begin{equation}
	\mathbf{\mu}=\mathbb{E}\mathbf{X}=\mathbb{E}\left(\mu_1,\ldots,\mu_k\right)^\intercal=\left(\mathbb{E}X_1,\ldots,\mathbb{E}X_k\right)^\intercal
\end{equation}
is the $k$-dimensional mean vector, and covariance matrix $\mathbf{\Sigma}\in\mathbb{R}^{k\times k}$ with
\begin{equation}
	\mathbf{\Sigma}\_{ij}=\mathbb{E}\left(X_i-\mu_i\right)\left(X_j-\mu_j\right)=\Cov(X_i,X_j)
\end{equation}
We also have that $\mathbf{\Sigma}\geq 0$ (positive semi-definite matrix)[^1].

Thus, the PDF of an MVN is defined as
\begin{equation}
f_X(x_1,\ldots,x_k)=\dfrac{1}{(2\pi)^{k/2}\vert\mathbf{\Sigma}\vert^{1/2}}\exp\left[\dfrac{1}{2}\left(\mathbf{x}-\mathbf{\mu}\right)^\intercal\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right]
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

#### Kernel functions
{: #kernel-func}
**Kernel function** is a real-valued function of two arguments
\begin{equation}
\kappa(\textbf{x},\textbf{x}')\in\mathbb{R},
\end{equation}
for $\textbf{x},\textbf{x}'\in\mathcal{X}$, which typically is symmetric (i.e., $\kappa(\textbf{x},\textbf{x}')=\kappa(\textbf{x}',\textbf{x})$), and nonnegative ($\kappa(\textbf{x},\textbf{x}')\geq0$). These are some examples of kernel functions.  


##### RBF kernels
Let $\mathcal{X}\subset\mathbb{R}^D$. For $\gamma>0$, a **Gaussian RBF kernels** or a **squared exponential kernel** (**SE kernel**) $\kappa: \mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is defined by
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\exp\left(-\frac{\Vert\textbf{x}-\textbf{x}'\Vert^2}{\gamma^2}\right)
\end{equation}
for $\textbf{x},\textbf{x}'\in\mathcal{X}$.


##### Mercer (positive definite) kernels
{: #mercer-kernels}
If the Gram matrix, defined by
\begin{equation}
\mathbf{K}=\begin{bmatrix}\kappa(\mathbf{x}\_1,\mathbf{x}\_1)&\ldots&\kappa(\mathbf{x}\_1,\mathbf{x}\_N) \\\\&\vdots&\\\\\kappa(\mathbf{x}\_N,\mathbf{x}\_1)&\ldots&\kappa(\mathbf{x}\_N,\mathbf{x}\_N)\end{bmatrix},
\end{equation}
is positive definite for any set $\left\\{x_i\right\\}\_{i=1}^N$, the kernel $\kappa$ is called a **Mercer kernel**, or **positive definite kernel**.


##### Linear kernels
{: #lin-kernels}
Deriving the feature vector implied by a kernel is only possible if the kernel is *Mercer*. However, deriving a kernel from a feature vector is easy. We have
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\phi\left(\textbf{x}\right)^\intercal\phi(\textbf{x}')
\end{equation}
If $\phi(\textbf{x})=\textbf{x}$, we obtain a simple kernel called **linear kernel**
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\textbf{x}^\intercal\textbf{x}'
\end{equation}


##### Matern kernels
{: #matern-kernels}
Let $\mathcal{X}\subset\mathbb{R}^D,\nu>0,\ell>0$. The **Matern kernel** $\kappa:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is defined by
\begin{equation}
\kappa(r)=\frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^{\nu}K_{\nu}\left(\frac{\sqrt{2\nu}r}{\ell}\right)\tag{2}\label{2}
\end{equation}
where $\Gamma$ is the *gamma function*; $r=\Vert\textbf{x}-\textbf{x}'\Vert$ for $\textbf{x},\textbf{x}'\in\mathcal{X}$ and $K_{\nu}$ is a *modified Bessel function*.  

As $\nu\to\infty$, \eqref{2} approaches the **SE kernel**
\begin{equation}
\lim_{\nu\to\infty}\kappa_\nu(r)=\exp\left(-\dfrac{r^2}{\ell}\right)
\end{equation}
where $\ell>0$.  

When $\nu$ is half-integer - i.e., $\nu=p+1/2$, for $p\geq0$, equation \eqref{2} can be reduced to a product of an exponential function and a polynomial of degree $p$, which can be written as
\begin{equation}
\kappa_{\nu=p+1/2}(r)=\exp\left(-\dfrac{\sqrt{2\nu}r}{\ell}\right)\dfrac{\Gamma(p+1)}{\Gamma(2p+1)}\sum_{i=0}^{p}\dfrac{(p+i)!}{i!(p-i)!}\left(\dfrac{\sqrt{8\nu}r}{\ell}\right)^{p-i}
\end{equation}

For an another special case, setting $\nu=1/2$ lets \eqref{2} become
\begin{equation}
\kappa_{\nu=1/2}=\exp\left(\dfrac{-r}{\ell}\right),
\end{equation}
which is known as **Laplace** or **exponential kernel**.  

#### Eigenfunction Analysis of Kernels
{: #eigenfunc-kernel}

##### Eigenfunctions
{: #eigenfunc}
A function $\phi(\cdot)$ that obeys the integral equation
\begin{equation}
\int \kappa(\textbf{x},\textbf{x}')\phi(\textbf{x})\,d\mu(\textbf{x})=\lambda\phi(\textbf{x}'),
\end{equation}
is called an **eigenfunction** of kernel $\kappa$ with eigenvalue $\lambda$ w.r.t [measure]({% post_url 2021-07-03-measure %}) $\mu$.  

**Mercer's theorem** allows us to express kernel $\kappa$ in terms of the eigenvalues and eigenfunctions.  

First, let us introduce an integral operator, $T_\kappa$, which is defined as
\begin{equation}
(T_\kappa f)(\textbf{x})=\int_\mathcal{X}\kappa(\textbf{x},\textbf{x}')f(\textbf{x}')\,d\mu(\textbf{x}'),
\end{equation}
where $\mu$ denotes a measure.

**Theorem** (*Mercer's theorem*)  
*Let $(\mathcal{X},\mu)$ be a finite measure space and $\kappa\in L_\infty(\mathcal{X}^2,\mu^2)$ be positive definite and $T_\kappa:L_2(\mathcal{X},\mu)\to L_2(\mathcal{X},\mu)$. Let $\phi_i\in L_2(\mathcal{X,\mu})$ be the normalized eigenfunctions of $T_\kappa$ associated with the eigenvalues $\lambda_i>0$. Then we have
\begin{equation}
\kappa(\textbf{x},\textbf{x}')=\sum_{i=0}^{\infty}\lambda_i\phi_i(\textbf{x})\phi_i(\textbf{x}')
\end{equation}
where the convergence is absolute and uniform over $\textbf{x},\textbf{x}'\in\mathcal{X}$.*

## Gaussian Process
{: #gp}
For Gaussian process, positive definite kernels serve as *covariance functions* of random function values, so they are also called *covariance kernels*.  

Let $\mathcal{X}$ be a nonempty set, $\kappa:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ be a positive definite kernel, and some real-valued function $\mu:\mathcal{X}\to\mathbb{R}$. Then a random function $f:\mathcal{X}\to\mathbb{R}$ is said to be a **Gaussian process** (**GP**) with mean function $\mu$ and covariance kernel $\kappa$, denoted by $\mathcal{GP(\mu,\kappa)}$, if the following holds:  
For any finite set $X=\\{\textbf{x}\_1,\ldots,\textbf{x}\_n\\}\subset\mathcal{X}$ of any size $n\in\mathbb{N}$, the random vector $f_X\in\mathbb{R}^n$,
\begin{equation}
f_X=\left(f(\textbf{x}\_1),\ldots,f(\textbf{x}\_n)\right)^\intercal\sim\mathcal{N}(\mu_X,\kappa_{XX})
\end{equation}
where $\mu_X=\left(\mu(\textbf{x}\_1),\ldots,\mu(\textbf{x}\_n)\right)^\intercal$ is the mean vector and $\kappa_{XX}=\left(\kappa(\textbf{x}\_i,\textbf{x}\_j)\right)\_{i,j=1}^n\in\mathbb{R}^{n\times n}$ is covariance matrix.  

The correpondence between **Gaussian process** $f\sim\mathcal{GP}(\mu,\kappa)$ and pairs $(\mu,\kappa)$ of mean function $\mu$ and positive definite kernel $\kappa$ is a one-to-one since from the definition above, Gaussian process $f$ implies the existence of a mean function $\mu:\mathcal{X}\to\mathbb{R}$ and a covariance kernel $\kappa:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$. And from  


### Gaussian Process Regression
{: #gpr}



## References
[1] C. E. Rasmussen & C. K. I. Williams. [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/), MIT Press, 2006  

[2] Motonobu Kanagawa, Philipp Hennig, Dino Sejdinovic, Bharath K. Sriperumbudur. [Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences](https://arxiv.org/abs/1807.02582)  

[3] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573)  

[4] Kevin P. Murphy. [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/book0.html), MIT Press, 2012  

[5] Peter I. Frazier. [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)  

[6] Martin Krasser. [Bayesian Optimization](https://krasserm.github.io/2018/03/21/bayesian-optimization/)  




## Footnotes
[^1]: The definition of covariance matrix $\mathbf{\Sigma}$ can be rewritten as
	\begin{equation}
	\mathbf{\Sigma}=\Cov(\mathbf{X},\mathbf{X})=\Var(\mathbf{X})
	\end{equation}
	Let $\mathbf{z}\in\mathbb{R}^k$, we have
	\begin{equation}
	\Var(\mathbf{z}^\intercal\mathbf{X})=\mathbf{z}^\intercal\Var(\mathbf{X})\mathbf{z}=\mathbf{z}^\intercal\mathbf{\Sigma}\mathbf{z}
	\end{equation}
	And since $\Var(\mathbf{z}^\intercal\mathbf{X})\geq0$, we also have that $\mathbf{z}^\intercal\mathbf{\Sigma}\mathbf{z}\geq0$, which proves that $\mathbf{\Sigma}$ is a positive semi-definite matrix.  

