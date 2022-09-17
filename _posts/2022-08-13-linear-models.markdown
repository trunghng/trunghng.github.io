---
layout: post
title:  "Linear models"
date:   2022-08-13 13:00:00 +0700
categories: artificial-intelligent machine-learning
tags: artificial-intelligent machine-learning linear-model
description: A note on lienar models
comments: true
---
> Materials were taken mostly from [Bishop's book](% post_url 2022-08-13-linear-models %}#bishops-book).
<!-- excerpt-end -->

- [Preliminaries](#preliminaries)
	- [Independence, basis, dimension](#ind-basis-dim)
		- [Linear independence](#lin-ind)
		- [Basis of a vector space](#basis)
- [Linear models for Regression](#lin-models-regression)
	- [Linear basis function models](#lin-basis-func-models)
		- [Least squares](#least-squares)
		- [Geometrical interpretation of least squares](#geo-least-squares)
		- [Regularized least squares](#reg-least-squares)
- [References](#references)
- [Footnotes](#footnotes)

## Preliminaries
{: #preliminaries}

### Independence, basis, dimension
{: #ind-basis-dim}

#### Linear independence
{: #lin-ind}
The sequence of vectors $\mathbf{x}\_1,\ldots,\mathbf{x}\_n$ is said to be **linearly independent** (or **independent**) if
\begin{equation}
c_1\mathbf{x}\_1+\ldots+c_n\mathbf{x}\_n=\mathbf{0}
\end{equation}
only when $c_1,\ldots,c_n$ are all zero.

Considering those $n$ vectors $\mathbf{x}\_1,\ldots,\mathbf{x}\_n$ as $n$ columns of a matrix $\mathbf{A}$
\begin{equation}
\mathbf{A}=\left[\begin{matrix}\vert&&\vert \\\\ \mathbf{x}\_1 & \ldots & \mathbf{x}\_n \\\\ \vert&&\vert\end{matrix}\right]
\end{equation}
we have that the columns of $\mathbf{A}$ are independent when
\begin{equation}
\mathbf{A}\mathbf{x}=\mathbf{0}\hspace{0.5cm}\Leftrightarrow\hspace{0.5cm}\mathbf{x}=\mathbf{0},
\end{equation}
or in other words, the rank of $\mathbf{A}$ is equal to the number of columns of $\mathbf{A}$.

#### Basis of a vector space
{: #basis}
We say that vectors $\mathbf{v}\_1,\ldots,\mathbf{v}\_k$ span a space $S$ when the space consists of all combinations of those vectors. Or in other words, any vector $\mathbf{u}\in S$ can be displayed as linear combination of $\mathbf{v}\_i$.  
In this case, $S$ is the smallest space containing those vectors.

A **basis** for a vector space $S$ is a sequence of vectors $\mathbf{v}\_1,\ldots,\mathbf{v}\_d$ having two properties:
<ul id='roman-list'>
	<li>$\mathbf{v}_1,\ldots,\mathbf{v}_d$ are independent</li>
	<li>$\mathbf{v}_1,\ldots,\mathbf{v}_d$ span $S$</li>
</ul>
In $S$, every basis for that space has the same number of vectors, which is the dimension of $S$. Therefore, there are exactly $n$ vectors in every basis for $\mathbb{R}^n$.

With that definition of a basis $\mathbf{v}\_1,\dots,\mathbf{v}\_d$ of $S$, for each vector $\mathbf{u}\in S$, there exists only one sequence $c_1,\ldots,c_d$ such that
\begin{equation}
\mathbf{u}=c_1\mathbf{v}\_1+\ldots+c_d\mathbf{v}\_d
\end{equation}

## Linear models for Regression
{: #lin-models-regression}
Regression refers to a problem of predicting the value of one or more continuous target variable $t$ given the value of a $D$-dimensional vector $\mathbf{x}$ of input variables.

### Linear basis function models
{: #lin-basis-func-models}
The simplest linear model used for regression tasks is **linear regression**, which is defined as a linear combination of the input variables
\begin{equation}
y(\mathbf{x},\mathbf{w})=w_0+w_1x_1+\ldots+w_Dx_D,\tag{1}\label{1}
\end{equation}
where $\mathbf{x}=(x_1,\ldots,x_D)^\intercal$ is the input variables, while $w_i$'s are the parameters parameterizing the space of linear function mapping from the input space $\mathcal{X}$ of $\mathbf{x}$ to $\mathcal{Y}$.

With the idea of spanning a space by its basis vectors, we can generalize it to establishing a function space by linear combinations of simpler basis functions. Or in other words, we can extend the class of models by instead using a linear combination of fixed nonlinear functions of the input variables $\mathbf{x}$, as
\begin{equation}
y(\mathbf{x},\mathbf{w})=w_0+w_1\phi_1(\mathbf{x})+\ldots+w_{M-1}\phi_{M-1}(\mathbf{x})=w_0+\sum_{i=1}^{M-1}w_i\phi_i(\mathbf{x}),\tag{2}\label{2}
\end{equation}
where $\phi_i(\mathbf{x})$'s are called the **basis functions**; $w_0$ is called a **bias parameter**. By letting $w_0$ be a coefficient corresponding to a dummy basis function $\phi_0(\mathbf{x})=1$, \eqref{2} can be written in a more convenient way
\begin{equation}
y(\mathbf{x},\mathbf{w})=\sum_{i=0}^{M-1}w_i\phi_i(\mathbf{x})=\mathbf{w}^\intercal\boldsymbol{\phi}(\mathbf{x}),\tag{3}\label{3}
\end{equation}
where $\mathbf{w}=(w_0,\ldots,w_{M-1})^\intercal$ and $\boldsymbol{\phi}=(\phi_0,\ldots,\phi_{M-1})^\intercal$, with $\phi_0(\cdot)=1$.

There are various choices of basis functions:
<ul id='number-list'>
	<li>
		<b>Polynomial basis</b>. Each basis function $\phi_i$ is a powers of a $1$-dimensional input $x$
		\begin{equation}
		\phi_i(x)=x^i
		\end{equation}
		An example of polynomial basis functions is illustrated as below
		<figure>
			<img src="/assets/images/2022-08-13/polynomial-basis.png" alt="polynomial basis" style="display: block; margin-left: auto; margin-right: auto;"/>
			<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: Example of polynomial basis functions. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/pattern-recognition-and-machine-learning-book/linear-regression-models/basis-funcs.py)</span></figcaption>
		</figure>
	</li>
	<li>
		<b>Gaussian basis function</b>. Each basis function $\phi_i$ is a Gaussian function of a $1$-dimensional input $x$
		\begin{equation}
		\phi_i(x)=\exp\left(-\frac{(x-\mu_i)^2}{2\sigma_i^2}\right)
		\end{equation}
		An example of Gaussian basis functions is illustrated as below
		<figure>
			<img src="/assets/images/2022-08-13/gaussian-basis.png" alt="Gaussian basis" style="display: block; margin-left: auto; margin-right: auto;"/>
			<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: Example of Gaussian basis functions. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/pattern-recognition-and-machine-learning-book/linear-regression-models/basis-funcs.py)</span></figcaption>
		</figure>
	</li>
	<li>
		<b>Sigmoidal basis function</b>. Each basis function $\phi_i$ is defined as
		\begin{equation}
		\phi_i(x)=\sigma\left(\frac{x-\mu_i}{\sigma_i}\right),
		\end{equation}
		where $\sigma(\cdot)$ is the logistic sigmoid function
		\begin{equation}
		\sigma(x)=\frac{1}{1+\exp(-x)}
		\end{equation}
		An example of sigmoidal basis functions is illustrated as below
		<figure>
			<img src="/assets/images/2022-08-13/sigmoidal-basis.png" alt="sigmoidal basis" style="display: block; margin-left: auto; margin-right: auto;"/>
			<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: Example of sigmoidal basis functions. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/pattern-recognition-and-machine-learning-book/linear-regression-models/basis-funcs.py)</span></figcaption>
		</figure>
	</li>
</ul>

#### Least squares
{: #least-squares}
Assume that the target variable $t$ and the inputs $\mathbf{x}$ is related via the equation
\begin{equation}
t=y(\mathbf{x},\mathbf{w})+\epsilon,
\end{equation}
where $\epsilon$ is an error term that captures random noise such that $\epsilon\sim\mathcal{N}(0,\sigma^2)$, which means the density of $\epsilon$ can be written as
\begin{equation}
p(\epsilon)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\epsilon^2}{2\sigma^2}\right),
\end{equation}
which implies that
\begin{equation}
p(t|\mathbf{x};\mathbf{w},\beta)=\sqrt{\frac{\beta}{2\pi}}\exp\left(-\frac{(t-y(\mathbf{x},\mathbf{w}))^2\beta}{2}\right),\tag{4}\label{4}
\end{equation}
where $\beta=1/\sigma^2$ is the precision of $\epsilon$, or
\begin{equation}
t|\mathbf{x};\mathbf{w},\beta\sim\mathcal{N}(y(\mathbf{x},\mathbf{w}),\beta^{-1})\tag{5}\label{5}
\end{equation}
Consider a data set of inputs $\mathbf{X}=\\{\mathbf{x}\_1,\ldots,\mathbf{x}\_N\\}$ with corresponding target values $\mathbf{t}=(t_1,\ldots,t_N)^\intercal$ and assume that these data points are drawn independently from the distribution \eqref{5}, we obtain the batch version of \eqref{4}, called the **likelihood function**, given as
\begin{align}
L(\mathbf{w},\beta)=p(\mathbf{t}|\mathbf{X};\mathbf{w},\beta)&=\prod_{i=1}^{N}p(t_i|\mathbf{x}\_i;\mathbf{w},\beta) \\\\ &=\prod_{i=1}^{N}\sqrt{\frac{\beta}{2\pi}}\exp\left(-\frac{(t_i-y(\mathbf{x}\_i,\mathbf{w}))^2\beta}{2}\right)
\end{align}
By maximum likelihood, we will be looking for values of $\mathbf{w}$ and $\beta$ that maximize the likelihood. We do this by considering maximizing a simpler likelihood, called **log likelihood**, denoted as $\ell(\mathbf{w},\beta)$, defined as
\begin{align}
\ell(\mathbf{w},\beta)=\log{L(\mathbf{w},\beta)}&=\log\prod_{i=1}^{N}\sqrt{\frac{\beta}{2\pi}}\exp\left(-\frac{(t_i-y(\mathbf{x}\_i,\mathbf{w}))^2\beta}{2}\right) \\\\ &=\sum_{i=1}^{N}\log\left[\sqrt{\frac{\beta}{2\pi}}\exp\left(-\frac{(t_i-y(\mathbf{x}\_i,\mathbf{w}))^2\beta}{2}\right)\right] \\\\ &=\frac{N}{2}\log\beta-\frac{N}{2}\log(2\pi)-\sum_{i=1}^{N}\frac{(t_i-y(\mathbf{x}\_i,\mathbf{w}))^2\beta}{2} \\\\ &=\frac{N}{2}\log\beta-\frac{N}{2}\log(2\pi)-\beta E_D(\mathbf{w})\tag{6}\label{6},
\end{align}
where $E_D(\mathbf{w})$ is the sum-of-squares error function, defined as
\begin{equation}
E_D(\mathbf{w})\doteq\frac{1}{2}\sum_{i=1}^{N}\left(t_i-y(\mathbf{x}\_i,\mathbf{w})\right)^2\tag{7}\label{7}
\end{equation}
Consider the gradient of \eqref{6} w.r.t $\mathbf{w}$, we have
\begin{align}
\nabla_\mathbf{w}\ell(\mathbf{w},\beta)&=\nabla_\mathbf{w}\left[\frac{N}{2}\log\beta-\frac{N}{2}\log(2\pi)-\beta E_D(\mathbf{w})\right] \\\\ &\propto\nabla_\mathbf{w}\frac{1}{2}\sum_{i=1}^{N}\big(t_i-y(\mathbf{x}\_i,\mathbf{w})\big)^2 \\\\ &=\nabla_\mathbf{w}\frac{1}{2}\sum_{i=1}^{N}\left(t_i-\mathbf{w}^\intercal\boldsymbol{\phi}\big(\mathbf{x}\_i\right)\big)^2 \\\\ &=\sum_{i=1}^{N}(t_i-\mathbf{w}^\intercal\boldsymbol{\phi}(\mathbf{x}\_i))\boldsymbol{\phi}(\mathbf{x}\_i)^\intercal
\end{align}
By gradient descent, letting this gradient to zero gives us
\begin{equation}
\sum_{i=1}^{N}t_i\boldsymbol{\phi}(\mathbf{x}\_i)^\intercal-\mathbf{w}^\intercal\sum_{i=1}^{N}\boldsymbol{\phi}(\mathbf{x}\_i)\boldsymbol{\phi}(\mathbf{x}\_i)^\intercal=0,
\end{equation}
which implies that
\begin{equation}
\mathbf{w}\_\text{ML}=\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\intercal\mathbf{t},\tag{8}\label{8}
\end{equation}
which is known as the **normal equations** for the least squares problem. In \eqref{8}, $\boldsymbol{\Phi}\in\mathbb{R}^{N\times M}$ is called the **design matrix**, whose elements are given by $\boldsymbol{\Phi}\_{ij}=\phi_j(\mathbf{x}\_i)$
\begin{equation}
\boldsymbol{\Phi}=\left[\begin{matrix}-\hspace{0.1cm}\boldsymbol{\phi}(\mathbf{x}\_1)\hspace{0.1cm}- \\\\ \hspace{0.1cm}\vdots\hspace{0.1cm} \\\\ -\hspace{0.1cm}\boldsymbol{\phi}(\mathbf{x}\_N)\hspace{0.1cm}-\end{matrix}\right]=\left[\begin{matrix}\phi_0(\mathbf{x}\_1)&\ldots&\phi_{M-1}(\mathbf{x}\_1) \\\\ \vdots&\ddots&\vdots \\\\ \phi_0(\mathbf{x}\_N)&\ldots&\phi_{M-1}(\mathbf{x}\_N)\end{matrix}\right],
\end{equation}
and the quantity
\begin{equation}
\boldsymbol{\Phi}^\dagger\doteq\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}^\intercal
\end{equation}
is called the **Moore-Penrose pseudoinverse** of the matrix $\boldsymbol{\Phi}$.

On the other hand, consider the gradient of \eqref{6} w.r.t $\beta$ and set it equal to zero, we obtain
\begin{equation}
\beta=\frac{N}{\sum_{i=1}^{N}\big(t_i-\mathbf{w}\_\text{ML}^\intercal\boldsymbol{\Phi}(\mathbf{x}\_i)\big)^2}
\end{equation}

#### Geometrical interpretation of least squares
{: #geo-least-squares}
As mentioned before, we have applied the idea of spanning a vector space by its basis vectors when constructing basis functions.

In particular, consider an $N$-dimensional space whose axes are given by $t_i$, which implies that
\begin{equation}
\mathbf{t}=(t_1,\ldots,t_N)^\intercal
\end{equation}
is a vector contained in the space.
<figure>
	<img src="/assets/images/2022-08-13/geo-least-squares.png" alt="geometry of least squares" style="display: block; margin-left: auto; margin-right: auto; width: 400px; height: 300px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 4</b>: Geometrical interpretation of the least-squares solution. The figure is taken from <span markdown="1">[Bishop's book](#bishops-book)</span></figcaption>
</figure>

Each basis function $\phi_j(\mathbf{x}\_i)$, evaluated at the $N$ data points, then can also be presented as a vector in the same space, denoted by $\boldsymbol{\varphi}\_j$, as illustrated in **Figure 4** above. Therefore, the design matrix $\boldsymbol{\Phi}$ can be represented as
\begin{equation}
\boldsymbol{\Phi}=\left[\begin{matrix}-\hspace{0.1cm}\boldsymbol{\phi}(\mathbf{x}\_1)\hspace{0.1cm}- \\\\ \hspace{0.1cm}\vdots\hspace{0.1cm} \\\\ -\hspace{0.1cm}\boldsymbol{\phi}(\mathbf{x}\_N)\hspace{0.1cm}-\end{matrix}\right]=\left[\begin{matrix}\vert&&\vert \\\\ \boldsymbol{\varphi}\_{0}&\ldots&\boldsymbol{\varphi}\_{M-1} \\\\ \vert&&\vert\end{matrix}\right]\tag{9}\label{9}
\end{equation}
When the number $M$ of basis functions is smaller than the number $N$ of data points, the $M$ vectors $\phi_j(\mathbf{x}\_i)$ will span a linear subspace $\mathcal{S}$ of $M$ dimensions.

We define $\mathbf{y}$ to be an $N$-dimensional vector whose the $i$-th element is given by $y(\mathbf{x}\_i,\mathbf{w})$
\begin{equation}
\mathbf{y}=\big(y(\mathbf{x}\_1,\mathbf{w}),\ldots,y(\mathbf{x}\_N,\mathbf{w})\big)^\intercal
\end{equation}
Since $\mathbf{y}$ is a linear combination of $\boldsymbol{\varphi}\_i$, then $\mathbf{y}\in\mathcal{S}$.
Then the sum-of-squares error \eqref{7} is exactly (with a factor of $1/2$) the squared Euclidean distance between $\mathbf{y}$ and $\mathbf{t}$. Therefore, the least square solution to $\mathbf{w}$ is the one that makes $\mathbf{y}$ closest to $\mathbf{t}$.

This solution corresponds to the orthogonal projection of $t$ onto the subspace $S$ spanned by $\boldsymbol{\varphi}\_i$, because we have that
\begin{align}
\mathbf{y}^\intercal(\mathbf{t}-\mathbf{y})&=\left(\boldsymbol{\Phi}\mathbf{w}\_\text{ML}\right)^\intercal\left(\mathbf{t}-\boldsymbol{\Phi}\mathbf{w}\_\text{ML}\right) \\\\ &=\left(\boldsymbol{\Phi}\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}\mathbf{t}\right)^\intercal\left(\mathbf{t}-\boldsymbol{\Phi}\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}\mathbf{t}\right) \\\\ &=\mathbf{t}^\intercal\boldsymbol{\Phi}\left(\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\right)^\intercal\boldsymbol{\Phi}^\intercal\mathbf{t}-\mathbf{t}^\intercal\boldsymbol{\Phi}\left(\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\right)^\intercal\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\boldsymbol{\Phi}\mathbf{t} \\\\ &=\mathbf{t}^\intercal\boldsymbol{\Phi}\left(\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\right)^\intercal\boldsymbol{\Phi}^\intercal\mathbf{t}-\mathbf{t}^\intercal\boldsymbol{\Phi}\left(\left(\boldsymbol{\Phi}^\intercal\boldsymbol{\Phi}\right)^{-1}\right)^\intercal\boldsymbol{\Phi}^\intercal\mathbf{t} \\\\ &=0,
\end{align}

#### Regularized least squares
{: #reg-least-squares}


## References
{: #references}
[1] <span id='bishops-book'>Christopher M. Bishop. [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732). Springer New York, NY.</span>

[2] Gilbert Strang. [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/).  

[3] MIT 18.06. [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/).

## Footnotes
{: #footnotes}