---
layout: post
title:  "Markov Chain"
date:   2021-06-19 22:27:00 +0700
categories: random-stuffs probability-statistics
comments: true
---
Since I have no idea how to begin with this post, why not just dive straight into details :P  

Markov chain is a stochastic process in which the random variables follow a special property called Markov.

### Markov property
A sequence of random variables $X_0, X_1, X_2, \dots$ taking values in the *state space* $S=${$1, 2,\dots, M$}. For all $n\geq0$,
\begin{equation}
P(X_{n+1}=j|X_n=i)=P(X_{n+1}=j|X_n=i,X_{n-1}=i_{n-1},X_{n-2}=i_{n-2},\dots,X_0=i_0)
\end{equation}
In other words, knowledge of the preceding state is all we need to determine the probability distribution of the current state.  

### Transition matrix
The quantity $P(X_{n+1}=j|X_n=i)$ is *transition probability* from state $i$ to $j$.  
If we denote that $q_{ij}=P(X_{n+1}=j|X_n=i)$ and let $Q=(q_{ij})$, which is a $M\times M$ matrix, there we have the *transition matrix* $Q$ of the chain.  
Therefore, each row of $Q$ is a conditional probability mass function (PMF) of $X_{n+1}$ given $X_n$. And hence, sum of its entries is 1.  

#### n-step transition probability
The n-step *transition probability* from $i$ to $j$ is the probability of being at $i$ and $n$ steps later being at $j$, and be denoted as $q_{ij}^{(n)}$,
\begin{equation}
q_{ij}^{(n)}=P(X_n=j|X_0=i)
\end{equation}
We have that
\begin{equation}
q_{ij}^{(2)}=\sum_{k}^{}q_{ik}q_{kj}
\end{equation}
since it has to go through an intermediary step $k$ to reach $j$ in 2 steps from $i$. It's easily seen that the right hand side is $Q_{ij}^2$. And by induction, we have that:
\begin{equation}
q_{ij}^{(n)}=Q_{ij}^{n}
\end{equation}
$Q^n$ is also called the *n-step transition matrix*.  

#### Marginal distribution of $X_n$
Let $t=(t_1,\dots,t_M)^T$, where $t_i=P(X_0=i)$. By the law of total probability (LOTP), we have that:
\begin{align}
P(X_n=j)&=\sum_{i=1}^{M}P(X_0=i)P(X_n=j|X_0=i) \\\\&=\sum_{i=1}^{M}t_iq_{ij}^{(n)}
\end{align}
or the marginal distribution of $X_n$ is given by $tQ^n$.

### Properties
- State $i$ of a Markov chain is defined as *recurrent* or *transient* depending upon whether or not the Markov chain will eventually return to it. Starting with *recurrent* state i, the chain will return to it with the probability of 1. Otherwise, it is *transient*. 
	- **Proposition**: Number of returns to *transient* state is distributed by *Geom($p$)*, with $p>0$ is the probability of never returning to $i$.
- A Markov chain is defined as *irreducible* if there exists a chain of steps between any $i,j$ that has positive probability. That is for any $i,j$, there is some $n>0,\in\mathbb{N}$ such that $Q^n_{ij}>0$. If not *irreducible*, it's called *reducible*
	- **Proposition**: *Irreducible* implies all states *recurrent*
- A state $i$ has *period* $k>0$ if $k$ is the greatest common divisor (gcd) of the possible numbers of steps it can take to return to $i$ when starting at $i$.
And thus, $k=gcd(n)$ such that $Q^n_{ii}>0$. $i$ is called *aperiodic* if $k_i=1$, and *periodic* otherwise. The chain itself is called *aperiodic* if all its states are *aperiodic*, and *periodic* otherwise.

### Stationary distribution
A vector $s=(s_1,\dots,s_M)^T$ such that $s_i\geq0$ and $\sum_{i}s_i=1$ is a *stationary distribution* for a Markov chain if
\begin{equation}
\sum_{i}s_iq_{ij}=s_j
\end{equation}
for all $j$, or equivalently $sQ=s$.  

**Theorem** (*Existence and uniqueness of stationary distribution*)  
&nbsp;&nbsp;&nbsp;&nbsp;Any *irreducible* Markov chain has a unique *stationary distribution*. In this distribution, every state has positive probability.  

The theorem is a consequence of a result from [*Perron-Frobenius theorem*](https://en.wikipedia.org/wiki/Perron–Frobenius_theorem).  

**Theorem** (*Convergence to stationary distribution*)  
&nbsp;&nbsp;&nbsp;&nbsp;Let $X_0,X_1,\dots$ be a Markov chain with *stationary distribution* $s$ and *transition matrix* $Q$, such that some power $Q^m$ has all entries positive (or in the other words, the chain is *irreducible* and *aperiodic*). Then $P(X_n=i)$ converges to $s_i$ as $n\rightarrow\infty$ (or $Q^n$ converges to a matrix in which each row is $s$).

**Theorem** (*Expected time to run*)  
&nbsp;&nbsp;&nbsp;&nbsp;Let $X_0,X_1,\dots$ be an *irreducible* Markov chain with *stationary distribution* $s$. Let $r_i$ be the expected time it takes the chain to return to $i$, given that it starts at $i$. Then $s_i=1/r_i$

### Reversibility
Let $Q=(q_{ij})$ be the *transition matrix* of a Markov chain. Suppose there is an $s=(s_1,\dots,s_M)^T$ with $s_i\geq0,\sum_{i}s_i=1$, such that
\begin{equation}
s_iq_{ij}=s_jq_{ji}
\end{equation}
for all states $i,j$. This equation is called *reversibility* or *detailed balance* condition. And if the condition holds, we say that the chain is *reversible* w.r.t $s$.  

**Proposition** (*Reversible implies stationary*)  
&nbsp;&nbsp;&nbsp;&nbsp;Suppose that $Q=(q_{ij})$ be the *transition matrix* of a Markov chain that is *reversible* w.r.t to an $s=(s_1,\dots,s_M)^T$ with with $s_i\geq0,\sum_{i}s_i=1$. Then $s$ is a *stationary distribution* of the chain. (*proof*:$\sum_{j}s_jq_{ji}=\sum_{j}s_iq_{ij}=s_i\sum_{j}q_{ij}=s_i$)  

**Proposition**  
&nbsp;&nbsp;&nbsp;&nbsp;If each column of $Q$ sum to 1, then the *uniform distribution* over all states $(1/M,\dots,1/M)$, is a *stationary distribution*. (This kind of matrix is called *doubly stochastic matrix*).

### Examples and application
- [*Finite-state machines*](https://en.wikipedia.org/wiki/Finite-state_machine), [*random walks*](https://en.wikipedia.org/wiki/Random_walk)
- Diced board games such as Ludo, Monopoly,...
- [*Google PageRank*](https://en.wikipedia.org/wiki/PageRank) - the heart of Google search
- Markov Decision Process (MDP), which is gonna be the content of next post.
- And various other applications.

#### Footnote:
- The Markov chain here is *time-homogeneous* Markov chain, in which the probability of any state transition is independent of time.
- This is more like intuitive and less formal definition of Markov chain, we will have more concrete definition with the help of *Measure theory* after the post about it.
- Well, it only matters where you are, not where you've been.

#### References:
1. Introduction to Probability - Joseph K. Blitzstein & Jessica Hwang
2. [Brillant's Markov chain](https://brilliant.org/wiki/markov-chains/)
