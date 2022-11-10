---
layout: post
title:  "Markov Chain"
date:   2021-06-19 22:27:00 +0700
tags: mathematics probability-statistics random-stuffs
description: A non-formal Markov chain definition
comments: true
eqn-number: true
---
> If we have to describe the defintition of **Markov chain** in one statement, it will be: "It only matters where you are, not where you've been".

<!-- excerpt-end -->
- [Markov Property](#markov-property)
- [Transition Matrix](#transition-matrix)
	- [n-step transition probability](#nstep-trans-prob)
- [Marginal distribution of $X_n$](#marginal-dist-xn)
- [Properties](#properties)
- [Stationary distribution](#stationary-distribution)
- [Reversibility](#reversibility)
- [Examples and Applications](#exp-app)
- [References](#references)
- [Footnotes](#footnotes)

## Markov Property
**Markov chain**[^1] is a stochastic process in which the random variables follow a special property called **Markov**.  

A sequence of random variables $X_0, X_1, X_2, \dots$ taking values in the **state space** $\mathcal{S}=${$1, 2,\dots, M$} such that for all $n\geq0$,
\begin{equation}
P(X_{n+1}=j|X_n=i)=P(X_{n+1}=j|X_n=i,X_{n-1}=i_{n-1},X_{n-2}=i_{n-2},\dots,X_0=i_0)
\end{equation}
In other words, knowledge of the preceding state is all we need to determine the probability distribution of the current state.  

## Transition Matrix
The quantity $P(X_{n+1}=j|X_n=i)$ is **transition probability** from state $i$ to $j$.  

If we denote that $q_{ij}=P(X_{n+1}=j|X_n=i)$ and let $Q$ $M\times M$ matrix, defined as
\begin{equation}
Q=\left[\begin{matrix}q_{11}&\ldots&q_{1M} \\\\ \vdots&\ddots&\vdots \\\\ q_{M1}&\ldots&q_{MM}\end{matrix}\right]
\end{equation}
The matrix $Q$ then is referred as the **transition matrix** of the chain.  

It is noticeable that each row of $Q$ is a conditional probability mass function (PMF) of $X_{n+1}$ given $X_n$. And hence, sum of its entries is 1.  

### $n$-step Transition Probability
{: #nstep-trans-prob}
The **$n$-step transition probability** from $i$ to $j$ is the probability of being at $i$ and $n$ steps later being at $j$, and be denoted as $q_{ij}^{(n)}$,
\begin{equation}
q_{ij}^{(n)}=P(X_n=j|X_0=i)
\end{equation}
We have that
\begin{equation}
q_{ij}^{(2)}=\sum_{k}^{}q_{ik}q_{kj}
\end{equation}
since it has to go through an intermediary step $k$ to reach $j$ in 2 steps from $i$. It is easily seen that the right hand side is $Q_{ij}^2$. And by induction, we have that:
\begin{equation}
q_{ij}^{(n)}=Q_{ij}^{n}
\end{equation}
$Q^n$ is also called the **$n$-step transition matrix**.  

### Marginal Distribution of $X_n$
{: #marginal-dist-xn}
Let $t=(t_1,\dots,t_M)^\text{T}$, where $t_i=P(X_0=i)$. By the **law of total probability** (LOTP), we have that:
\begin{equation}
P(X_n=j)=\sum_{i=1}^{M}P(X_0=i)P(X_n=j|X_0=i)=\sum_{i=1}^{M}t_iq_{ij}^{(n)},
\end{equation}
which implies that the marginal distribution of $X_n$ is given by $tQ^n$.

## Properties
<ul id='roman-list'>
	<li>
		State $i$ of a Markov chain is defined as <b>recurrent</b> or <b>transient</b> depending upon whether or not the Markov chain will eventually return to it. Starting with <b>recurrent</b> state $i$, the chain will return to it with the probability of $1$. Otherwise, it is <b>transient</b>.<br>
		<b>Proposition</b>: Number of returns to <b>transient</b> state is distributed by $\text{Geom}(p)$, with $p>0$ is the probability of never returning to $i$.
	</li>
	<li>
		A Markov chain is defined as <b>irreducible</b> if there exists a chain of steps between any $i,j$ that has positive probability. That is for any $i,j$, there is some $n>0,\in\mathbb{N}$ such that $Q^n_{ij}>0$. If not <b>irreducible</b>, the chain is instead referred as <b>reducible</b>.<br>
		<b>Proposition</b>: <b>irreducible</b> implies all states <b>recurrent</b>.
	</li>
	<li>
		A state $i$ has <b>period</b> $k>0$ if
		\begin{equation}
		k=\text{gcd}(n),
		\end{equation}
		where $n$ is possible number of steps it can take to return to $i$ when starting at $i$, or $Q^n_{ii}>0$.<br>
		State $i$ is known as <b>aperiodic</b> if $k_i=1$, and <b>periodic</b> otherwise. The chain itself is called <b>aperiodic</b> if all its states are <b>aperiodic</b>, and <b>periodic</b> otherwise.
	</li>
</ul>

## Stationary Distribution
A vector $s=(s_1,\dots,s_M)^\text{T}$ such that $s_i\geq0$ and $\sum_{i}s_i=1$ is a **stationary distribution** for a Markov chain if
\begin{equation}
\sum_{i}s_iq_{ij}=s_j
\end{equation}
for all $j$, or equivalently $sQ=s$.  

**Theorem** (*Existence and uniqueness of stationary distribution*)  
*Any irreducible Markov chain has a unique stationary distribution. In this distribution, every state has positive probability.*  

The theorem is a consequence of a result from **Perron-Frobenius theorem**.  

**Theorem** (*Convergence to stationary distribution*)  
*Let $X_0,X_1,\dots$ be a Markov chain with stationary distribution $s$ and transition matrix $Q$, such that some power $Q^m$ has all entries positive (or in the other words, the chain is irreducible and aperiodic). Then
\begin{equation}
P(X_n=i)\to s_i
\end{equation}
as $n\rightarrow\infty$ (or $Q^n$ converges to a matrix in which each row is $s$)*.  

**Theorem** (*Expected time to run*)  
*Let $X_0,X_1,\dots$ be an irreducible Markov chain with stationary distribution $s$. Let $r_i$ be the expected time it takes the chain to return to $i$, given that it starts at $i$. Then*
\begin{equation}
s_i=\frac{1}{r_i}
\end{equation}

## Reversibility
Let $Q=(q_{ij})$ denote transition matrix of a Markov chain. Suppose there is an $s=(s_1,\dots,s_M)^\text{T}$ with $s_i\geq0,\sum_{i}s_i=1$, such that
\begin{equation}
s_iq_{ij}=s_jq_{ji}
\end{equation}
for all states $i,j$. This equation is called **reversibility** or **detailed balance** condition. And if the condition holds, we say that the chain is **reversible** w.r.t $s$.  

**Proposition** (*Reversible implies stationary*)  
Suppose that $Q=(q_{ij})$ be the transition matrix of a Markov chain that is reversible w.r.t to an $s=(s_1,\dots,s_M)^\text{T}$ with $s_i\geq0,\sum_{i}s_i=1$. Then $s$ is a stationary distribution of the chain.

**Proof**  
We have that
\begin{equation}
\sum_{j}s_jq_{ji}=\sum_{j}s_iq_{ij}=s_i\sum_{j}q_{ij}=s_i
\end{equation}  

**Proposition**  
If each column of $Q$ sum to $1$, then the **Uniform distribution** over all states $(1/M,\dots,1/M)$, is a stationary distribution (this kind of matrix is called **doubly stochastic matrix**).

## Examples and Applications
{: #exp-app}
- [**Finite-state machines**](https://en.wikipedia.org/wiki/Finite-state_machine), [**random walks**](https://en.wikipedia.org/wiki/Random_walk)
- Diced board games such as Ludo, Monopoly,...
- [**Google PageRank**](https://en.wikipedia.org/wiki/PageRank) - the heart of Google search
- [**Markov Decision Process** (MDP)]({% post_url 2021-06-27-mdp-bellman-eqn %}).
- And various other applications.

## References
[1] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573).  

[2] [Brillant's Markov chain](https://brilliant.org/wiki/markov-chains/).  

[3] [Perron-Frobenius theorem](https://en.wikipedia.org/wiki/Perron–Frobenius_theorem).

## Footnotes
[^1]: The Markov chain here is **time-homogeneous** Markov chain, in which the probability of any state transition is independent of time.
