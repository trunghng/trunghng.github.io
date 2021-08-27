---
layout: post
title:  "Monte Carlo methods in Reinforcement Learning"
date:   2021-08-21 13:03:00 +0700
categories: [artificial-intelligent, reinforcement-learning]
tags: artificial-intelligent reinforcement-learning monte-carlo
description: Monte Carlo methods for solving Reinforcement Learning problems
comments: true
---
> Recall that in the previous post, [**Dynamic Programming Algorithms For Solving Markov Decision Processes**]({% post_url 2021-07-25-dp-in-mdp %}), we made an assumption about the complete knowledge of the environment. With **Monte Carlo** methods, we only require *experience* - sample sequences of states, actions, and rewards from simulated or real interaction with an environment.

<!-- excerpt-end -->
- [Monte Carlo methods](#mc-methods)
- [Monte Carlo methods in Reinforcement Learning](#mc-rl)
	- [Monte Carlo Prediction](#mc-prediction)
		- [First-visit MC vs. every-visit MC](#first-mc-every-mc)
	- [Monte Carlo Control](#mc-control)
	- [Monte Carlo Control without Exploring Starts](#mc-control-wo-es)
- [References](#references)
- [Footnotes](#footnotes)



## Monte Carlo Methods[^1]
{: #mc-methods}
**Monte Carlo**, named after a casino in Monaco, simulates complex probabilistic events using simple random events, such as tossing a pair of dice to simulate the casino's overall business model.

<figure>
	<img src="/assets/images/2021-08-21/mc-pi.gif" alt="monte carlo method" width="480" height="360px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: Using Monte Carlo method to approximate the value of $\pi$</figcaption>
</figure><br/>

Monte Carlo methods have been used in several different tasks:
1. Simulating a system and its probability distribution $\pi(x)$
\begin{equation}
x\sim\pi(x)
\end{equation}
2. Estimating a quantity through Monte Carlo integration
\begin{equation}
c=\mathbb{E}\_\pi\left[f(x)\right]=\int\pi(x)f(x)\,dx
\end{equation}
3. Optimizing a target function to find its modes (maxima or minima)
\begin{equation}
x^\*=\arg\max\pi(x)
\end{equation}
4. Learning a parameters from a training set to optimize some loss functions, such as the maximum likelihood estimation from a set of examples $\\{x_i,i=1,2,\dots,M\\}$
\begin{equation}
\Theta^\*=\arg\max\sum_{i=1}^{M}\log p(x_i;\Theta)
\end{equation}
5. Visualizing the energy landscape of a target function


## Monte Carlo Methods in Reinforcement Learning
{: #mc-rl}
Monte Carlo (MC) methods are ways of solving the reinforcement learning problem based on averaging sample returns. Here, we define Monte Carlo methods only for episodic tasks. Or in other words, they learn from complete episodes of experience.

### Monte Carlo Prediction[^2]
{: #mc-prediction}
Since the value of a state $v_\pi(s)=\mathbb{E}\_\pi\left[G_t|S_t=s\right]$ is defined as the expectation of the return when the process is started from the given state $s$, an obvious way of estimating this value from experience is to compute observed mean returns after visits to that state. As more returns are observed, the average should converge to the expected value. This is an instance of the so-called *Monte Carlo method*.  

In particular, suppose we wish to estimate $v_\pi(s)$ given a set of episodes obtained by following $\pi$ and passing through $s$. Each time state $s$ appears in an episode, we call it a *visit* to $s$. There are two types of Monte Carlo methods:
- *First-visit MC method*
	- estimates $v_\pi(s)$ as the average of the returns that have followed the *first visit* to $s$.
	- We call the first time $s$ is visited in an episode the *first visit* to $s$.
- *Every-visit MC method*
	- estimates $v_\pi(s)$ as the average of the returns that have followed all visits to to $s$.  

The sample mean return for state $s$ is:
\begin{equation}
v_\pi(s)=\dfrac{\sum_{t=1}^{T}𝟙\left(S_t=s\right)G_t}{\sum_{t=1}^{T}𝟙\left(S_t=s\right)},
\end{equation}
where $𝟙(\cdot)$ is an indicator function. In the case of *first-visit MC*, $𝟙\left(S_t=s\right)$ returns $1$ only in the first time $s$ is encountered in an episode. And for *every-visit MC*, $𝟙\left(S_t=s\right)$ gives value of $1$ every time $s$ is visited.  

Here is the pseudocode of the *first-visit MC prediction*, for estimating $V\approx v_\pi$
<figure>
	<img src="/assets/images/2021-08-21/mc-prediction.png" alt="iterative policy evaluation pseudocode" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### First-visit MC vs. every-visit MC
{: #first-mc-every-mc}
Both methods converge to $v_\pi(s)$ as the number of visits (or first visits) to $s$ goes to infinity. Each average is itself an unbiased estimate, and the standard deviation of its error falls as $\frac{1}{\sqrt{n}}$, where $n$ is the number of returns averaged.

<figure>
	<img src="/assets/images/2021-08-21/first-visit-every-visit.png" alt="first-visit MC vs every-visit MC" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: Summary of Statistical Results comparing first-visit and every-visit MC method</figcaption>
</figure><br/>

### Monte Carlo Control[^3]
{: #mc-control}
When model is not available, it is particular useful to estimate *action values* rather than *state values* (which alone are insufficient to determine a policy). We must explicitly estimate the value of each action in order for the values to be useful in suggesting a policy. Thus, one of our primary goals for MC methods is to estimate $q_\*$. To achieve this, we first consider the policy evaluation problem for action values.  

Similar to when using MC method to estimate $v_\pi(s)$, we can use both first-visit MC and every-visit MC to approximate the value of $q_\pi(s,a)$. The only thing we need to keep in mind is, in this case, we work with visits to a state-action pair rather than to a state. Likewise, we define two types of MC methods for estimating $q_\pi(s,a)$:
- *First-visit MC method*
	- estimates $q_\pi(s,a)$ as the average of the returns following the first time in each episode that the state $s$ was visited and the action $a$ was selected
- *Every-visit MC method*
	- estimates $q_\pi(s,a)$ as the average of the returns that have followed all the visits to state-action pair $(s,a)$.

(TODO)  

To learn the optimal policy by MC, we apply the idea of [GPI]({% post_url 2021-07-25-dp-in-mdp %}#gpi):
\begin{equation}
\pi_0\overset{\small \text{E}}{\rightarrow}q_{\pi_0}\overset{\small \text{I}}{\rightarrow}\pi_1\overset{\small \text{E}}{\rightarrow}q_{\pi_1}\overset{\small \text{I}}{\rightarrow}\pi_2\overset{\small \text{E}}{\rightarrow}\dots\overset{\small \text{I}}{\rightarrow}\pi_\*\overset{\small \text{E}}{\rightarrow}q_\*
\end{equation}
In particular,
1. *Policy evaluation* (denoted as $\overset{\small\text{E}}{\rightarrow}$): estimates action value function $q_\pi(s,a)$ using the episode generated from $s, a$, following by current policy $\pi$
\begin{equation}
q_\pi(s,a)=\dfrac{\sum_{t=1}^{T}𝟙\left(S_t=s,A_t=a\right)G_t}{\sum_{t=1}^{T}𝟙\left(S_t=s,A_t=a\right)}
\end{equation}
2. *Policy improvement* (denoted as $\overset{\small\text{I}}{\rightarrow}$): makes the policy *greedy* with the current value function (action value function in this case)
\begin{equation}
\pi(s)\doteq\arg\max_{a\in\mathcal{A(s)}} q(s,a)
\end{equation}
The policy improvement can be done by constructing each $\pi_{k+1}$ as the greedy policy w.r.t $q_{\pi_k}$ because
\begin{align}
q_{\pi_k}\left(s,\pi_{k+1}(s)\right)&=q_{\pi_k}\left(s,\arg\max_a q_{\pi_k}(s,a)\right) \\\\ &=\max_a q_{\pi_k}(s,a) \\\\ &\geq q_{\pi_k}\left(s,\pi_k(s)\right) \\\\ &\geq v_{\pi_k}(s)
\end{align}
Therefore, by [policy improvement theorem]({% post_url 2021-07-25-dp-in-mdp %}#policy-improvement), we have that $\pi_{k+1}\geq\pi_k$.  
<figure>
	<img src="/assets/images/2021-08-21/gpi.png" alt="GPI" width="150" height="150px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: MC policy iteration</figcaption>
</figure><br/>
To solve this problem with Monte Carlo policy iteration, in the 1998 version of ''*Reinforcement Learning: An Introduction*", authors of the book introduced **Monte Carlo ES** (MCES), for Monte Carlo with *Exploring Starts*.  

In MCES, value function is approximated by simulated returns and a greedy policy is selected at each iteration. Although MCES does not converge to any suboptimal policy, the convergence to optimal fixed point is still an open question. For solutions in particular settings, you can check out some results like Tsitsiklis (2002), Liu (2020), Chen (2018).  
Down below is the pseudocode of the Monte Carlo ES.
<figure>
	<img src="/assets/images/2021-08-21/mces.png" alt="monte carlo es pseudocode" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

### Monte Carlo Control without Exploring Starts
{: #mc-control-wo-es}

## References
[1] Reinforcement Learning: An Introduction - Richard S. Sutton & Andrew G. Barto  

[2] Monte Carlo Methods - Adrian Barbu & Song-Chun Zhu  

[3] [UCL course on RL](https://www.davidsilver.uk/teaching/) - David Silver  

[4] Algorithms for Reinforcement Learning - Csaba Szepesvári  

[5] Singh, S.P., Sutton, R.S. [Reinforcement learning with replacing eligibility traces](https://doi.org/10.1007/BF00114726). Mach Learn 22, 123–158 (1996)  

[6] John N. Tsitsiklis. [On the Convergence of Optimistic Policy Iteration](https://www.mit.edu/~jnt/Papers/J089-02-jnt-optimistic.pdf). Journal of Machine Learning Research 3 (2002) 59–72  

[7] Jun Liu. [On the Convergence of Reinforcement Learning with Monte Carlo Exploring Starts](https://arxiv.org/abs/2007.10916) (2020)  

[8] Yuanlong Chen. [On the convergence of optimistic policy iteration for stochastic shortest path problem](https://arxiv.org/abs/1808.08763) (2018)  

[9] 

## Footnotes
[^1]: We are gonna talk about Monte Carlo methods in more detail in another post.
[^2]: A prediction task in RL is where we are given a policy and our goal is to measure how well it performs.
[^3]: In contrast to prediction, a control task in RL is where the policy is not fixed, and our goal is to find the optimal policy.
