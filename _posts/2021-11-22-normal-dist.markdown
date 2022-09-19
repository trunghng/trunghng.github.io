---
layout: post
title:  "Normal Distribution"
date:   2021-11-22 14:46:00 +0700
categories: mathematics probability-statistics
tags: mathematics probability-statistics normal-distribution
description: Normal Distribution
comments: true
---
> A note on Normal distribution.
<!-- excerpt-end -->

- [Gaussian (Normal) Distribution](#gauss-dist)
	- [Standard Normal](#std-norm)
- [Multivariate Normal Distribution](#mvn)
	- [Bivariate Normal](#bvn)
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
\varphi(x)=\dfrac{1}{\sqrt{2\pi}}e^{-z^2/2}
\end{equation}
\begin{equation}
\Phi(x)=\int_{-\infty}^{x}\varphi(t)\,dt=\int_{-\infty}^{x}\dfrac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt
\end{equation}
Below are some visualizations of Normal distribution.

<figure>
	<img src="/assets/images/2021-11-22/normal.png" alt="normal distribution" style="display: block; margin-left: auto; margin-right: auto; width:  900px; height: 380px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: 10K normally distributed data points (5K each plot) were plotted as vertical bars on x-axis. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/bayes-optimization/gauss-dist.py)</span></figcaption>
</figure><br/>

## Multivariate Normal Distribution
{: #mvn}
A $k$-dimensional random vector $\mathbf{X}=\left(X_1,\dots,X_k\right)^T$ is said to have a **Multivariate Normal (MVN)** distribution if every linear combination of the $X_i$ has a Normal distribution. Which means
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
	\mathbf{\Sigma}\_{ij}=\mathbb{E}\left(X_i-\mu_i\right)\left(X_j-\mu_j\right)=\Cov(X_i,X_j)
\end{equation}
We also have that $\mathbf{\Sigma}\geq 0$ (positive semi-definite matrix)[^1].

Thus, the PDF of an MVN is defined as
\begin{equation}
f_X(x_1,\ldots,x_k)=\dfrac{1}{(2\pi)^{k/2}\vert\mathbf{\Sigma}\vert^{1/2}}\exp\left[\dfrac{1}{2}\left(\mathbf{x}-\mathbf{\mu}\right)^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\mathbf{\mu})\right]
\end{equation}
With this idea, *Standard Normal* distribution in multi-dimensional case can be defined as a Gaussian with mean $\mathbf{\mu}=0$ (here $0$ is an $k$-dimensional vector) and identity covariance matrix $\mathbf{\Sigma}=\mathbf{I}\_{k\times k}$.

### Bivariate Normal
{: #bvn}
When the number of dimensions in $\mathbf{X}$, $k=2$, this special case of MVN is called the **Bivariate Normal (BVN)**.

An example of an BVN, $\mathcal{N}\left(\left[\begin{smallmatrix}0\\\\0\end{smallmatrix}\right],\left[\begin{smallmatrix}1&0.5\\\\0.8&1\end{smallmatrix}\right]\right)$, is shown as following.  

<figure>
	<img src="/assets/images/2021-11-22/bvn.png" alt="monte carlo method" style="display: block; margin-left: auto; margin-right: auto; width: 750px; height: 350px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The PDF of $\mathcal{N}\left(\left[\begin{smallmatrix}0\\0\end{smallmatrix}\right],\left[\begin{smallmatrix}1&0.5\\0.8&1\end{smallmatrix}\right]\right)$. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/bayes-optimization/mvn.py)</span></figcaption>
</figure><br/>

## References
{: #references}
[1] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573). 

## Footnotes
{: #footnotes}
[^1]: The definition of covariance matrix $\mathbf{\Sigma}$ can be rewritten as
	\begin{equation}
	\mathbf{\Sigma}=\Cov(\mathbf{X},\mathbf{X})=\Var(\mathbf{X})
	\end{equation}
	Let $\mathbf{z}\in\mathbb{R}^k$, we have
	\begin{equation}
	\Var(\mathbf{z}^T\mathbf{X})=\mathbf{z}^T\Var(\mathbf{X})\mathbf{z}=\mathbf{z}^T\mathbf{\Sigma}\mathbf{z}
	\end{equation}
	And since $\Var(\mathbf{z}^T\mathbf{X})\geq0$, we also have that $\mathbf{z}^T\mathbf{\Sigma}\mathbf{z}\geq0$, which proves that $\mathbf{\Sigma}$ is a positive semi-definite matrix.