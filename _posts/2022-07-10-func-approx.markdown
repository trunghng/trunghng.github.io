---
layout: post
title:  "Function Approximation"
date:   2022-07-10 15:26:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning function-approximation
description: Function approximation
comments: true
---
> 
<!-- excerpt-end -->
- [On-policy Methods](#on-policy-methods)
	- [Value-function Approximation](#value-func-approx)
	- [The Prediction Objective](#pred-obj)
	- [Gradient-based algorithms](#grad-algs)
		- [Stochastic-gradient](#stochastic-grad)
		- [Semi-gradient](#on-policy-semi-grad)
	- [Linear Function Approximation](#lin-func-approx)
		- [Linear Methods](#lin-methods)
		- [Feature Construction](#feature-cons)
			- [Polinomials](#polinomials)
			- [Fourier Basis](#fourier)
			- [Coarse Coding](#coarse-coding)
			- [Tile Coding](#tile-coding)
			- [Radial Basis Functions](#rbf)
	- [Least-Squares TD](#lstd)
	- [Episodic Semi-gradient Control](#ep-semi-grad-control)
	- [Semi-gradient n-step Sarsa](#semi-grad-n-step-sarsa)
- [Off-policy Methods](#off-policy-methods)
	- [Semi-gradient](#off-policy-semi-grad)
	- [Gradient-TD](#grad-td)
	- [Emphatic-TD](#em-td)


## On-policy Methods
{: #on-policy-methods}
So far in the series, we have gone through tabular methods, which are used to solve problems with small state and action spaces. For larger spaces, rather than getting the exact solutions, we now have to approximate the value of them. To start, we begin with on-policy approximation methods.

### Value-function Approximation
{: #value-func-approx}
All of the prediction methods so far have been described as updates to an estimated value function that shift its value at particular states toward a "backed-up value" (or *update target*) for that state
\begin{equation}
s\mapsto u,
\end{equation}
where $s$ is the state updated and $u$ is the update target that $s$'s estimated value is shifted toward. 

For example, 
- the MC update for value prediction is: $S_t\mapsto G_t$.
- the TD(0) update for value prediction is: $S_t\mapsto R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)$.
- the $n$-step TD update is: $S_t\mapsto G_{t:t+n}$.
- and in the DP, policy-evaluation update, $s\mapsto\mathbb{E}\big[R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)\vert S_t=s\big]$, an arbitrary $s$ is updated. 

Each update $s\mapsto u$ can be viewed as example of the desired input-output behavior of the value function. And when the outputs are numbers, like $u$, we call the process **function approximation**.

### The Prediction Objective
{: #pred-obj}
In constrast to tabular case, where the solution of value function could be found equal to the true value function exactly, and an update at one state did not affect the others, with function approximation, it is imposible to find the exact value function of all states. And moreover, an update at one state also affects many others. 

Hence, it is necessary to specify a state distribution $\mu(s)\geq0,\sum_s\mu(s)=1$, representing how much we care about the error (the difference between the approximate value $\hat{v}(s,\mathbf{w})$ and the true value $v_\pi(s)$) in each state $s$. Weighting this over the state space $\mathcal{S}$ by $\mu$, we obtain a natural objective function, called the *Mean Squared Value Error*, denoted as $\overline{\text{VE}}$:
\begin{equation}
\overline{\text{VE}}(\mathbf{w})\doteq\sum_{s\in\mathcal{S}}\mu(s)\Big[v_\pi(s)-\hat{v}(s,\mathbf{w})\Big]^2
\end{equation}
The distribution $\mu(s)$ is usually chosen as the fraction of time spent in $s$ (number of time $s$ visited divived by total amount of visits). Under on-policy training this is called the *on-policy distribution*.  

- In continuing tasks, the on-policy distribution is the stationary distribution under $\pi$.  
- In episodic tasks, the on-policy distribution depends on how the initial states are chosen.
	- Let $h(s)$ denote the probability that an episode begins in each state $s$, and let $\eta(s)$ denote the number of time steps spent, on average, in state $s$ in a single episode
	\begin{equation}
	\eta(s)=h(s)+\sum_\bar{s}\eta(\bar{s})\sum_a\pi(a\vert\bar{s})p(s\vert\bar{s},a),\hspace{1cm}\forall s\in\mathcal{S}
	\end{equation}
	This system of equation can be solved for the expected number of visits $\eta(s)$. The on-policy distribution is then
	\begin{equation}
	\mu(s)=\frac{\eta(s)}{\sum_{s'}\eta(s')},\hspace{1cm}\forall s\in\mathcal{S}
	\end{equation}

### Gradient-based algorithms
{: #grad-algs}
To solve the least squares problem, we are going to use a popular method, named **Gradient descent**. 

Say, consider a differentiable function $J(\mathbf{w})$ of parameter vector $\mathbf{w}$.  

The gradient of $J(\mathbf{w})$ w.r.t $\mathbf{w}$ is defined to be
\begin{equation}
\nabla_{\mathbf{w}}J(\mathbf{w})=\left(\begin{smallmatrix}\dfrac{\partial J(\mathbf{w})}{\partial\mathbf{w}\_1} \\\\ \vdots \\\\ \dfrac{\partial J(\mathbf{w})}{\partial\mathbf{w}\_n}\end{smallmatrix}\right)
\end{equation}
The idea of Gradient descent is to minimize the objective function $J(\mathbf{w})$, we repeatly move $\mathbf{w}$ in the direction of steepest decrease of $J$, which is the direction of negative gradient $-\nabla_\mathbf{w}J(\mathbf{w})$. 

Thus, we have the update rule of Gradient descent:
\begin{equation}
\mathbf{w}\leftarrow\mathbf{w}-\dfrac{1}{2}\alpha\nabla_\mathbf{w}J(\mathbf{w}),
\end{equation}
where $\alpha$ is a positive step-size parameter.

#### Stochastic-gradient
{: #stochastic-grad}
Apply gradient descent to our problem, which is we have to find the minimization of
\begin{equation}
\overline{\text{VE}}(\mathbf{w})=\sum_{s\in\mathcal{S}}\mu(s)\Big[v_\pi(s)-\hat{v}(s,\mathbf{w})\Big]^2
\end{equation}
Since $\mu(s)$ is the state distribution over state space $\mathcal{S}$, we can rewrite $\overline{\text{VE}}$ as
\begin{equation}
\overline{\text{VE}}(\mathbf{w})=\mathbb{E}\_{s\sim\mu}\Big[v_\pi(s)-\hat{v}(s,\mathbf{w})\Big]^2
\end{equation}
By the update we have defined earlier, in each step, we need to decrease $\mathbf{w}$ by an amount of
\begin{equation}
\Delta\mathbf{w}=-\dfrac{1}{2}\alpha\nabla_\mathbf{w}\overline{\text{VE}}(\mathbf{w})=\alpha\mathbb{E}\Big[v_\pi(s)-\hat{v}(s,\mathbf{w})\Big]\nabla_\mathbf{w}\hat{v}(s,\mathbf{w})
\end{equation}
Using **Stochastic Gradient descent (SGD)**, and since the Monte Carlo target $G_t$ by definition is an unbiased estimate of $v_\pi(S_t)$ , we sample the gradient:
\begin{equation}
\Delta\mathbf{w}=\alpha(G_t-\hat{v}(S_t,\mathbf{w}))\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w})
\end{equation}
which gives us pseudocode of the algorithm:
<figure>
	<img src="/assets/images/2022-07-10/sgd_mc.png" alt="SGD Monte Carlo" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### Semi-gradient
{: #on-policy-semi-grad}
If instead of using MC target $G_t$, we use the bootstrapping targets such as $n$-step return $G_{t:t+n}$ or the DP target $\sum_{a,s',r}\pi(a\vert S_t)p(s',r\vert S_t,a)\left[r+\gamma\hat{v}(s',\mathbf{w}\_t)\right]$, which all depend on the current value of the weight vector $\mathbf{w}\_t$, and then implies that they will be biased, and will not produce a true gradient-descent method. 

Such methods are called **semi-gradient** since they include only a part of the gradient.
<figure>
	<img src="/assets/images/2022-07-10/semi_gd.png" alt="Semi-gradient TD(0)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

### Linear Function Approximation
{: #lin-func-approx}
One of the most crucial special cases of function approximation is that in which the approximate function, $\hat{v}(\cdot,\mathbf{w})$, is a linear function of the weight vector, $\mathbf{w}$.

#### Linear Methods
{: #lin-methods}

#### Feature Construction
{: #feature-cons}

##### Polinomials
{: #polinomials}

##### Fourier Basis
{: #fourier}
**Fourier series** is applied widely in Maths to approximate a periodic function[^1]. For example:

<figure>
	<img src="/assets/images/2022-07-10/fourier_series.gif" alt="Fourier series visualization" width="480" height="360px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: Four partial sums (Fourier series) of lengths 1, 2, 3, and 4 terms, showing how the approximation to a square wave improves as the number of terms increases: where $f_1(x)=\frac{4\sin\theta}{\pi},f_2(x)=\frac{4\sin3\theta}{3\pi},f_3=\frac{4\sin5\theta}{5\pi}$ and $f_4(x)=\frac{4\sin7\theta}{7\pi}$. The code can be found <span markdown="1">[here](https://github.com/trunghng/maths-visualization/blob/main/fourier-series/fourier_series.py)</span></figcaption>
</figure><br/>


## References
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] Konidaris, G. & Osentoski, S. & Thomas, P.. [Value Function Approximation in Reinforcement Learning Using the Fourier Basis](#https://dl.acm.org/doi/10.5555/2900423.2900483). AAAI Conference on Artificial Intelligence, North America, aug. 2011. 

[3] Deepmind x UCL. [Reinforcement Learning Lecture Series 2021](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)


## Footnotes
[^1]: A function $f$ is periodic with period $T$ if
	\begin{equation}
	f(x+T)=f(x),\forall x
	\end{equation}