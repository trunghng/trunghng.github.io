---
layout: post
title:  "Temporal-Difference Learning"
date:   2022-04-08 16:55:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning td-learning n-step-td q-learning my-rl
description: Temporal-Difference Learning, Q-learning
comments: true
---
> So far in this [series](/tag/my-rl), we have gone through the ideas of [**dynamic programming** (DP)]({% post_url 2021-07-25-dp-in-mdp %}) and [**Monte Carlo**]({% post_url 2021-08-21-monte-carlo-in-rl %}). What will happen if we combine these ideas together? **Temporal-deffirence (TD) learning** is our answer.

<!-- excerpt-end -->
- [TD(0)](#td0)
	- [TD Prediction](#td-prediction)
		- [Adventages over MC & DP](#adv-over-mc-dp)
		- [Optimality of TD(0)](#opt-td0)
	- [TD Control](#td-control)
		- [Sarsa](#sarsa)
		- [Q-learining](#q-learning)
			- [Example: Cliffwalking - Sarsa vs Q-learning](#eg-cliffwalking)
		- [Expected Sarsa](#exp-sarsa)
		- [Double Q-learning](#double-q-learning)
			- [Maximization Bias](#max-bias)
			- [A Solution](#sol)
- [$\boldsymbol{n}$-step TD](#n-step-td)
	- [$\boldsymbol{n}$-step TD Prediction](#n-step-td-prediction)
- [References](#references)
- [Footnotes](#footnotes)

## TD(0)
{: #td0}
As usual, we approach this new method in the prediction problem.

### TD Prediction
Borrowing the idea of Monte Carlo, TD methods learn from episodes of experience to solve the [prediction problem]({% post_url 2021-08-21-monte-carlo-in-rl %}#fn:2). The simplest TD method is **TD(0)** (or **one-step TD**)[^1], which has the update form:
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\right]\tag{1}\label{1},
\end{equation}
where $\alpha>0$ is step size of the update. Here is pseudocode of the TD(0) method
<figure>
	<img src="/assets/images/2022-04-08/td0.png" alt="TD(0)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>
Recall that in [Monte Carlo method]({% post_url 2021-08-21-monte-carlo-in-rl %}#mc-prediction), or even in its trivial form, **constant-$\alpha$ MC**, which has the update form:
\begin{equation}
V(S_t)\leftarrow V(S_t)+\alpha\left[G_t-V(S_t)\right]\tag{2}\label{2},
\end{equation}
we have to wait until the end of the episode, when the return $G_t$ is determined. However, with TD(0), we can do the update immediately in the next time step $t+1$.  

As we can see in \eqref{1} and \eqref{2}, both TD and MC updates look ahead to a sample successor state (or state-action pair), use the value of the successor and the corresponding reward in order to update the value of the current state (or state-action pair). This kind of updates is called *sample update*, which differs from *expected update* used by DP methods in that they are based on a single sample successor rather than on a complete distribution of all possible successors.

Other than the sampling of Monte Carlo, TD methods also use the bootstrapping of DP. Because similar to [DP]({% post_url 2021-07-25-dp-in-mdp %}#policy-evaluation), TD(0) is also a bootstrapping method, since the target in its update is $R_{t+1}+\gamma V(S_{t+1})$.  

The quantity inside bracket in \eqref{1} is called *TD error*, denoted as $\delta$:
\begin{equation}
\delta_t\doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)
\end{equation}
If the array $V$ does not change during the episode (as in MC), then the MC error can be written as a sum of TD errors
\begin{align}
G_t-V(S_t)&=R_{t+1}+\gamma G_{t+1}-V(S_t)+\gamma V(S_{t+1})-\gamma V(S_{t+1}) \\\\ &=\delta_t+\gamma\left(G_{t+1}-V(S_{t+1})\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\left(G_{t+2}-V(S_{t+2})\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\dots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\left(G_T-V(S_T)\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\dots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}(0-0) \\\\ &=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
\end{align}

#### Adventages over MC & DP
{: #adv-over-mc-dp}
With how TD is established, these are some advantages of its over MC and DP:
- Only experience is required.
- Can be fully incremental:
	- Can make update before knowing the final outcome.
	- Requires less memory.
	- Requires less peak computation.  


TD(0) does converge to $v_\pi$, in the mean for a sufficient small $\alpha$, and with probability of $1$ if $\alpha$ decreases according to the *stochastic approximation condition*
\begin{equation}
\sum_{n=1}^{\infty}\alpha_n(a)=\infty\hspace{1cm}\text{and}\hspace{1cm}\sum_{n=1}^{\infty}\alpha_n^2(a)<\infty,
\end{equation}
where $\alpha_n(a)$ denote the step-size parameter used to process the reward received after the $n$-th selection of action $a$.  

#### Optimality of TD(0)
{: #opt-td0}
Under batch training, TD(0) converges to the optimal maximum likelihood estimate. The convergence and optimality proofs can be found in this [paper](#td-convergence).
<figure>
	<img src="/assets/images/2022-04-08/random_walk_batch_updating.png" alt="TD(0) vs constant-alpha MC" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: Performance of TD(0) and constant-$\alpha$ MC under batch training on the random walk task. The code can be found <span markdown="1">[here](https://github.com/trunghng/reinforcement-learning-an-introduction-imp/blob/main/chapter-6/random-walk.py)</span></figcaption>
</figure>

### TD Control
We begin solving the control problem with an on-policy TD method. Recall that in on-policy methods, we evaluate or improve the policy $\pi$ used to make decision.

#### Sarsa
As mentioned in [MC methods]({% post_url 2021-08-21-monte-carlo-in-rl %}#mc-est-action-value), when the model is not available, we have to learn an action-value function rather than a state-value function. Or in other words, we need to estimate $q_\pi(s,a)$ for the current policy $\pi$ and $\forall s,a$. Thus, instead of considering transitions from state to state in order to learn the value of states, we now take transitions from state-action pair to state-action pair into account so as to learn the value of state-action pairs.  

Similarly, we've got the TD update for the action-value function case:
\begin{equation}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\right]\tag{3}\label{3}
\end{equation}
This update is done after every transition from a nonterminal state $S_t$ to its successor $S_{t+1}$
\begin{equation}
\left(S_t,A_t,R_{t+1},S_{t+1},A_{t+1}\right)
\end{equation}
And if $S_{t+1}$ is terminal (i.e., $S_{t+1}=S_T$), then $Q(S_{t+1},A_{t+1})=0$. The name **Sarsa** of the method is taken based on acronym of the quintuple.  

As usual when working on on-policy control problem, we apply the idea of [GPI]({% post_url 2021-07-25-dp-in-mdp %}#gpi):
\begin{equation}
\pi_0\overset{\small \text{E}}{\rightarrow}q_{\pi_0}\overset{\small \text{I}}{\rightarrow}\pi_1\overset{\small \text{E}}{\rightarrow}q_{\pi_1}\overset{\small \text{I}}{\rightarrow}\pi_2\overset{\small \text{E}}{\rightarrow}\dots\overset{\small \text{I}}{\rightarrow}\pi_\*\overset{\small \text{E}}{\rightarrow}q_\*
\end{equation}
However this time, instead, we use it with TD methods. Which is, we continually estimate $q_\pi$ for the behavior policy $\pi$, and at the same time change $\pi$ toward greediness w.r.t $q_\pi$. That gives us the following pseudocode of the Sarsa control algorithm
<figure>
	<img src="/assets/images/2022-04-08/sarsa.png" alt="Sarsa" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### Q-learning
{: #q-learning}
We now turn our move to an off-policy method, which evaluates or improves a policy different from that used to generate the data.  
The method we are talking about is called **Q-learning**, which was first introduced by [Watkin](#q-learning-watkins), in which the update on $Q$-value has the form:
\begin{equation}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right]\tag{4}\label{4}
\end{equation}
In this case, the learned action-value function, $Q$, directly approximates optimal action-value function $q_*$, independent of the policy being followed. Down below is pseudocode of the $Q$-learning.
<figure>
	<img src="/assets/images/2022-04-08/q-learning.png" alt="Q-learning" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

##### Example: Cliffwalking - Sarsa vs Q-learning
{: #eg-cliffwalking}

#### Expected Sarsa
{: #exp-sarsa}
In the update \eqref{4} of Q-learning, rather than taking the maximum over next state-action pairs, if we use the expected value to consider how likely each action is under the current policy. That means, we instead have the following update rule for $Q$-value:
\begin{align}
Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\mathbb{E}\_\pi\big[Q(S_{t+1},A_{t+1}\vert S_{t+1})\big]-Q(S_t,A_t)\Big] \\\\ &\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_a\pi(a|S_{t+1})Q(S_{t+1}|a)-Q(S_t,A_t)\Big]
\end{align}
However, given the next state, $S_{t+1}$, this algorithms move *deterministically* in the same direction as Sarsa moves in *expectation*. Thus, this method is also called **Expected Sarsa**.  

Expected Sarsa performs better than Sarsa since it eliminates the variance due to the randomization in selecting $A_{t+1}$. Which also means that it takes expected Sarsa more resource than Sarsa.

#### Double Q-learning
{: #double-q-learning}

##### Maximization Bias
{: #max-bias}
Consider a set of $M$ random variables $X=\\{X_1,\dots,X_M\\}$. Say that we are interested in maximizing expected value of the r.v.s in $X$:
\begin{equation}
\max_{i=1,\dots,M}\mathbb{E}(X_i)
\end{equation}
This value can be approximated by constructing approximations for $\mathbb{E}(X_i),\forall i$. Let $S=\bigcup_{i=1}^{M}S_i$ denote a set of samples, where $S_i$ is the subset containing samples for the variables $X_i$, and assume that the samples in $S_i$ are i.i.d. Unbiased estimates for the expected values can be obtained by computing the sample average for each variable:
\begin{equation}
\mathbb{E}(X_i)=\mathbb{E}(\mu_i)\approx\mu_i(S)\doteq\frac{1}{\vert S_i\vert}\sum_{s\in S_i}s,
\end{equation}
where $\mu_i$ is an estimator for variable $X_i$. This approximation is unbiased since every sample $s\in S_i$ is an unbiased estimate for the value of $\mathbb{E}(X_i)$.



##### A Solution
The reason why maximization bias happens is we are using the same samples to decide which action is the best (highest reward one) and also to estimate its action-value.

Double Q-learning is a variant of Q-learning[^2].

<figure>
	<img src="/assets/images/2022-04-08/double-q-learning.png" alt="Double Q-learning" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

## $\boldsymbol{n}$-step TD
{: #n-step-td}

### $\boldsymbol{n}$-step TD Prediction
{: #n-step-td-prediction}


## References
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] <span id='td-convergence'>Sutton, R.S. [Learning to predict by the methods of temporal differences](https://doi.org/10.1007/BF00115009). Mach Learn 3, 9–44 (1988).</span>  

[3] <span id='q-learning-watkins'>Chris Watkins. [Learning from Delayed Rewards](https://www.researchgate.net/publication/33784417_Learning_From_Delayed_Rewards). PhD Thesis (1989)</span>  

[4] Hado Hasselt. [Double Q-learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html). NIPS 2010


## Footnotes
[^1]: It is a special case of [n-step TD](#n-step-td) and TD($\lambda$).
[^2]: Another popular variant of Q-learning is [Deep Q-learning](https://www.nature.com/articles/nature14236), which was introduced by Deepmind in 2015. We're gonna talk about it in the post of Function approximation.