---
layout: post
title:  "Temporal-Difference Learning"
date:   2022-04-08 16:55:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning td-learning n-step-td q-learning my-rl
description: Temporal-Difference Learning, Q-learning
comments: true
---
> So far in this [series](/tag/my-rl), we have gone through the ideas of [**dynamic programming** (DP)]({% post_url 2021-07-25-dp-in-mdp %}) and [**Monte Carlo**]({% post_url 2021-08-21-monte-carlo-in-rl %}). What will happen if we combine these ideas together? **Temporal-difference (TD) learning** is our answer.

<!-- excerpt-end -->
- [TD(0)](#td0)
	- [TD Prediction](#td-prediction)
		- [Advantages over MC & DP](#adv-over-mc-dp)
		- [Optimality of TD(0)](#opt-td0)
	- [TD Control](#td-control)
		- [Sarsa](#sarsa)
		- [Q-learning](#q-learning)
			- [Example: Cliffwalking - Sarsa vs Q-learning](#eg-cliffwalking)
		- [Expected Sarsa](#exp-sarsa)
		- [Double Q-learning](#double-q-learning)
			- [Maximization Bias](#max-bias)
			- [A Solution](#sol)
- [$n$-step TD](#n-step-td)
	- [$n$-step TD Prediction](#n-step-td-prediction)
		- [Example: Random Walk](#eg-random-walk)
	- [$n$-step TD Control](#n-step-td-control)
		- [$n$-step Sarsa](#n-step-sarsa)
	- [Off-policy n-step TD](#off-policy-n-step-td)
		- [$n$-step TD with Importance Sampling](#n-step-td-is)
		- [$n$-step Tree Backup](#n-step-tree-backup)
		- [$n$-step $Q(\\sigma)$](#n-step-q-sigma)
- [References](#references)
- [Footnotes](#footnotes)

## TD(0)
{: #td0}
As usual, we approach this new method by considering the prediction problem.

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

The quantity inside bracket in \eqref{1} is called <span id='td_error'>**TD error**</span>, denoted as $\delta$:
\begin{equation}
\delta_t\doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)
\end{equation}
If the array $V$ does not change during the episode (as in MC), then the MC error can be written as a sum of TD errors
\begin{align}
G_t-V(S_t)&=R_{t+1}+\gamma G_{t+1}-V(S_t)+\gamma V(S_{t+1})-\gamma V(S_{t+1}) \\\\ &=\delta_t+\gamma\left(G_{t+1}-V(S_{t+1})\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\left(G_{t+2}-V(S_{t+2})\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\dots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\left(G_T-V(S_T)\right) \\\\ &=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\dots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}(0-0) \\\\ &=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
\end{align}

#### Advantages over MC & DP
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
This update is done after every transition from a non-terminal state $S_t$ to its successor $S_{t+1}$
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
The method we are talking about is called **Q-learning**, which was first introduced by [Watkins](#q-learning-watkins), in which the update on $Q$-value has the form:
\begin{equation}
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\left[R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t)\right]\tag{4}\label{4}
\end{equation}
In this case, the learned action-value function, $Q$, directly approximates optimal action-value function $q_*$, independent of the policy being followed. Down below is pseudocode of the Q-learning.
<figure>
	<img src="/assets/images/2022-04-08/q-learning.png" alt="Q-learning" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

##### Example: Cliffwalking - Sarsa vs Q-learning
{: #eg-cliffwalking}
(This example is taken from *Example 6.6, Reinforcement Learning: An Introduction book*.)
<figure>
	<img src="/assets/images/2022-04-08/cliff-walking-eg.png" alt="Cliff Walking example" style="display: block; margin-left: auto; margin-right: auto; width: 500px"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>
Say that we have an agent in a gridworld, which is an undiscounted, episodic task described by the above image. Start and goal states are denoted as "S" and "G" respectively. Agent can take up, down, left or right action. All the actions lead to a reward of $-1$, except for cliff region, into which stepping gives a reward of $-100$. We will be solving this problem with Q-learning and Sarsa with $\varepsilon$-greedy action selection, for $\varepsilon=0.1$.

**Solution code**  
The source code can be found [here](https://github.com/trunghng/reinforcement-learning-an-introduction-imp/blob/main/chapter-6/cliff_walking.py).  

<button type="button" class="collapsible" id="codeP1">Click to show the code</button>
<div class="codePanel" id="codeP1data" markdown="1">
<br>
We begin by importing necessary packages we will be using

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```
Our first step is to define the environment, gridworld with a cliff, which is constructed by height, width, cliff region, start state, goal state, actions and rewards.
```python
class GridWorld:

    def __init__(self, height, width, start_state, goal_state, cliff):
        '''
        Initialization function

        Params
        ------
        height: int
            gridworld's height
        width: int
            gridworld's width
        start_state: [int, int]
            gridworld's start state
        goal_state: [int, int]
            gridworld's goal state
        cliff: list<[int, int]>
            gridworld's cliff region
    	'''
        self.height = height
        self.width = width
        self.start_state = start_state
        self.goal_state = goal_state
        self.cliff = cliff
        self.actions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self.rewards = {'cliff': -100, 'non-cliff': -1}
```
The gridworld also needs some helper functions. `is_terminal()` function checks whether the current state is the goal state; `take_action()` takes an state and action as inputs and returns next state and corresponding reward while `get_action_idx()` gives us the index of action from action list. Putting all these functions inside `GridWorld`'s body, we have:
```python
    def is_terminal(self, state):
        '''
        Whether state @state is the goal state

        Params
        ------
        state: [int, int]
            current state
        '''
        return state == self.goal_state


    def take_action(self, state, action):
        '''
        Take action @action at state @state

        Params
        ------
        state: [int, int]
            current state
        action: (int, int)
            action taken

        Return
        ------
        (next_state, reward): ([int, int], int)
            a tuple of next state and reward
        '''
        next_state = [state[0] + action[0], state[1] + action[1]]
        next_state = [max(0, next_state[0]), max(0, next_state[1])]
        next_state = [min(self.height - 1, next_state[0]), min(self.width - 1, next_state[1])]
        if next_state in self.cliff:
            reward = self.rewards['cliff']
            next_state = self.start_state
        else:
            reward = self.rewards['non-cliff']
        return next_state, reward


    def get_action_idx(self, action):
        '''
        Get index of action in action list

        Params
        ------
        action: (int, int)
            action
        '''
        return self.actions.index(action)
```
Next, we define the $\varepsilon$-greedy function used by our methods in `epsilon_greedy()` function.
```python
def epsilon_greedy(grid_world, epsilon, Q, state):
    '''
    Choose action according to epsilon-greedy policy

    Params:
    -------
    grid_world: GridWorld
    epsilon: float
    Q: np.ndarray
        action-value function
    state: [int, int]
        current state

    Return
    ------
    action: (int, int)
    '''
    if np.random.binomial(1, epsilon):
        action_idx = np.random.randint(len(grid_world.actions))
        action = grid_world.actions[action_idx]
    else:
        values = Q[state[0], state[1], :]
        action_idx = np.random.choice([action_ for action_, value_ 
            in enumerate(values) if value_ == np.max(values)])
        action = grid_world.actions[action_idx]
    return action
```
It's time for our main course, Q-learning and Sarsa.
```python
def q_learning(Q, grid_world, epsilon, alpha, gamma):
    '''
    Q-learning

    Params
    ------
    Q: np.ndarray
        action-value function
    grid_world: GridWorld
    epsilon: float
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    state = grid_world.start_state
    rewards = 0

    while not grid_world.is_terminal(state):
        action = epsilon_greedy(grid_world, epsilon, Q, state)
        next_state, reward = grid_world.take_action(state, action)
        rewards += reward
        action_idx = grid_world.get_action_idx(action)
        Q[state[0], state[1], action_idx] += alpha * (reward + gamma * \
            np.max(Q[next_state[0], next_state[1], :]) - Q[state[0], state[1], action_idx])
        state = next_state

    return rewards

def sarsa(Q, grid_world, epsilon, alpha, gamma):
    '''
    Sarsa

    Params
    ------
    Q: np.ndarray
        action-value function
    grid_world: GridWorld
    epsilon: float
    alpha: float
        step size
    gamma: float
        discount factor
    '''
    state = grid_world.start_state
    action = epsilon_greedy(grid_world, epsilon, Q, state)
    rewards = 0

    while not grid_world.is_terminal(state):
        next_state, reward = grid_world.take_action(state, action)
        rewards += reward
        next_action = epsilon_greedy(grid_world, epsilon, Q, next_state)
        action_idx = grid_world.get_action_idx(action)
        next_action_idx = grid_world.get_action_idx(next_action)
        Q[state[0], state[1], action_idx] += alpha * (reward + gamma * Q[next_state[0], \
            next_state[1], next_action_idx] - Q[state[0], state[1], action_idx])
        state = next_state
        action = next_action

    return rewards
```

And lastly, wrapping everything together in the main function, we have
```python
if __name__ == '__main__':
    height = 4
    width = 13
    start_state = [3, 0]
    goal_state = [3, 12]
    cliff = [[3, x] for x in range(1, 12)]
    grid_world = GridWorld(height, width, start_state, goal_state, cliff)
    n_runs = 50
    n_eps = 500
    epsilon = 0.1
    alpha = 0.5
    gamma = 1
    Q = np.zeros((height, width, len(grid_world.actions)))
    rewards_q_learning = np.zeros(n_eps)
    rewards_sarsa = np.zeros(n_eps)

    for _ in tqdm(range(n_runs)):
        Q_q_learning = Q.copy()
        Q_sarsa = Q.copy()

        for ep in range(n_eps):
            rewards_q_learning[ep] += q_learning(Q_q_learning, grid_world, epsilon, alpha, gamma)
            rewards_sarsa[ep] += sarsa(Q_sarsa, grid_world, epsilon, alpha, gamma)

    rewards_q_learning /= n_runs
    rewards_sarsa /= n_runs

    plt.plot(rewards_q_learning, label='Q-Learning')
    plt.plot(rewards_sarsa, label='Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('./cliff_walking.png')
    plt.close()
```
</div>  

This is our result after completing running the code.
<figure>
	<img src="/assets/images/2022-04-08/cliff_walking.png" alt="Q-learning vs Sarsa on Cliff walking" style="display: block; margin-left: auto; margin-right: auto; width: 500px"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### Expected Sarsa
{: #exp-sarsa}
In the update \eqref{4} of Q-learning, rather than taking the maximum over next state-action pairs, we use the expected value to consider how likely each action is under the current policy. That means, we instead have the following update rule for $Q$-value:
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
\max_{i=1,\dots,M}\mathbb{E}(X_i)\tag{5}\label{5}
\end{equation}
This value can be approximated by constructing approximations for $\mathbb{E}(X_i)$ for all $i$. Let
\begin{equation}
S=\bigcup_{i=1}^{M}S_i
\end{equation}
denote a set of samples, where $S_i$ is the subset containing samples for the variables $X_i$, and assume that the samples in $S_i$ are i.i.d. Unbiased estimates for the expected values can be obtained by computing the sample average for each variable:
\begin{equation}
\mathbb{E}(X_i)=\mathbb{E}(\mu_i)\approx\mu_i(S)\doteq\frac{1}{\vert S_i\vert}\sum_{s\in S_i}s,
\end{equation}
where $\mu_i$ is an estimator for variable $X_i$. This approximation is unbiased since every sample $s\in S_i$ is an unbiased estimate for the value of $\mathbb{E}(X_i)$. Thus, \eqref{5} can be approximated by:
\begin{equation}
\max_{i=1,\dots,M}\mathbb{E}(X_i)=\max_{i=1,\dots,M}\mathbb{E}(\mu_i)\approx\max_{i=1,\dots,M}\mu_i(S)\tag{6}\label{6}
\end{equation}
Let $f_i$, $F_i$ denote the PDF and CDF of $X_i$ and $f_i^\mu, F_i^\mu$ denote the PDF and CDF of $\mu_i$ respectively. Hence we have that
\begin{align}
\mathbb{E}(X_i)&=\int_{-\infty}^{\infty}x f_i(x)\,dx;\hspace{0.5cm}F_i(x)=P(X_i\leq x)=\int_{-\infty}^{\infty}f_i(x)\,dx \\\\ \mathbb{E}(\mu_i)&=\int_{-\infty}^{\infty}x f_i^\mu(x)\,dx;\hspace{0.5cm}F_i^\mu(x)=P(\mu_i\leq x)=\int_{-\infty}^{\infty}f_i^\mu(x)\,dx
\end{align}
With these notations, considering the maximal estimator $\mu_i$, which is distributed by some PDF $f_{\max}^{\mu}$, we have:
\begin{align}
F_{\max}^{\mu}&\doteq P(\max_i \mu_i\leq x) \\\\ &=P(\mu_1\leq x;\dots;\mu_M\leq x) \\\\ &=\prod_{i=1}^{M}P(\mu_i\leq x) \\\\ &=\prod_{i=1}^{M}F_i^\mu(x)
\end{align}
The value $\max_i\mu_i(S)$ is an unbiased estimate of $\mathbb{E}(\max_i\mu_i)$, which is given by
\begin{align}
\mathbb{E}\left(\max_i\mu_i\right) &=\int_{-\infty}^{\infty}x f_{\max}^{\mu}(x)\,dx \\\\ &=\int_{-\infty}^{\infty}x\frac{d}{dx}\left(\prod_{i=1}^{M}F_i^\mu(x)\right)\,dx \\\\ &=\sum_{i=1}^M\int_{-\infty}^{\infty}f_i^\mu(x)\prod_{j\neq i}^{M}F_i^\mu(x)\,dx
\end{align}
However, as can be seen in \eqref{5}, the order of expectation and maximization is the other way around. This leads to the result that $\max_i\mu_i(S)$ is a biased estimate of $\max_i\mathbb{E}(X_i)$

##### A Solution
{: #sol}
The reason why maximization bias happens is we are using the same samples to decide which action is the best (highest reward one) and also to estimate its action-value. To overcome this situation, Hasselt (2010) proposed an alternative method that uses two set of estimators instead, $\mu^A=\\{\mu_1^A,\dots,\mu_M^A\\}$ and $\mu^B=\\{\mu_1^B,\dots,\mu_M^B\\}$. The method is thus also called **double estimators**. 

Specifically, we use these two sets to learn two independent estimates, called $Q^A$ and $Q^B$, each is an estimate of the true value $q(a)$, for all $a\in\mathcal{A}$. 

<figure>
	<img src="/assets/images/2022-04-08/double-q-learning.png" alt="Double Q-learning" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>


## $\boldsymbol{n}$-step TD
{: #n-step-td}
From the definition of *one-step TD*, we can formalize the idea into a more general, **n-step TD**. Once again, first off, we will be considering the prediction problem.


### $\boldsymbol{n}$-step TD Prediction
{: #n-step-td-prediction}
Recall that in *one-step TD*, the update is based on the next reward, bootstrapping[^2] from the value of the state at one step later. In particular,
the target of the update is $R_{t+1}+\gamma V_t(S_{t+1})$, which we are going to denote as $G_{t:t+1}$, or *one-step return*:
\begin{equation}
G_{t:t+1}\doteq R_{t+1}+\gamma V_t(S_{t+1})
\end{equation}
where $V_t:\mathcal{S}\to\mathbb{R}$ is the estimate at time step $t$ of $v_\pi$. Thus, rather than taking into account one step later, in *two-step TD*, it makes sense to consider the rewards in two steps further, combined with the value function of the state at two step later. In other words, the target of two-step update is the *two-step return*:
\begin{equation}
G_{t:t+2}\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2 V_{t+1}(S_{t+2})
\end{equation}
Similarly, the target of $n$-step update is the *$n$-step return*:
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n V_{t+n-1}(S_{t+n})
\end{equation}
for all $n,t$ such that $n\geq 1$ and $0\leq t\<T-n$. If $t+n\geq T$, then all the missing terms are taken as zero, and the *n-step return* defined to be equal to the full return:
\begin{equation}
G_{t:t+n}=G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\dots+\gamma^{T-t-1}R_T,\tag{7}\label{7}
\end{equation}
which is the target of the Monte Carlo update.  

Hence, the **$\boldsymbol{n}$-step TD** method can be defined as:
\begin{equation}
V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha\left[G_{t:t+n}-V_{t+n-1}(S_t)\right],
\end{equation}
for $0\leq t\<T$, while the values for all other states remain unchanged: $V_{t+n}(s)=V_{t+n-1}(s),\forall s\neq S_t$. Pseudocode of the algorithm is given right below.
<figure>
	<img src="/assets/images/2022-04-08/n-step-td.png" alt="n-step TD" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

From \eqref{7} combined with this definition of *$n$-step TD* method, it is easily seen that by changing the value of $n$ from $1$ to $\infty$, we obtain a corresponding spectrum ranging from *one-step TD method* to *Monte Carlo method*.
<figure>
	<img src="/assets/images/2022-04-08/n-step-td-diagram.png" alt="Backup diagram of n-step TD" style="display: block; margin-left: auto; margin-right: auto; width: 450px; height: 370px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The backup diagram of $n$-step TD methods</figcaption>
</figure>

#### Example: Random Walk
{: #eg-random-walk}
(This example is taken from *Example 7.1, Reinforcement Learning: An Introduction book*; the random process image is created based on the figure from [Singd & Sutton](#random_walk)).  

Suppose we have a random process as following
<figure>
	<img src="/assets/images/2022-04-08/random_process.png" alt="Random process" style="display: block; margin-left: auto; margin-right: auto; width: 620px; height: 120px"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>
Specifically, the reward is zero everywhere except the transitions into terminal states: the transition from State 2 to State 1 (with reward of $-1$) and the transition from State 20 to State 21 (with reward of $1$). The discount factor, $\gamma$, is $1$. The initial value estimates are $0$ for all states. We will implement $n$-step TD method for $n\in\\{1,2,4,\dots,512\\}$ and step size $\alpha\in\\{0,0.2,0.4,\dots,1\\}$. The walk starts at State 10. 

**Solution code**  
The source code can be found [here](https://github.com/trunghng/reinforcement-learning-an-introduction-imp/blob/main/chapter-7/random_walk.py).  

<button type="button" class="collapsible" id="codeP2">Click to show the code</button>
<div class="codePanel" id="codeP2data" markdown="1">
<br>
As usual, we need these packages for our implementation.

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```
First off, we need to define our environment, the random walk process. The `is_terminal()` function is used to check whether the state considered is a terminal state, while the `take_action()` function itself returns the next state and corresponding reward given the current state and the action taken.
```python
class RandomWalk:
    '''
    Random walk environment
    '''

    def __init__(self, n_states, start_state):
        self.n_states = n_states
        self.states = np.arange(1, n_states + 1)
        self.start_state = start_state
        self.end_states = [0, n_states + 1]
        self.actions = [-1, 1]
        self.action_prob = 0.5
        self.rewards = [-1, 0, 1]


    def is_terminal(self, state):
        '''
        Whether state @state is an end state

        Params
        ------
        state: int
            current state
        '''
        return state in self.end_states


    def take_action(self, state, action):
        '''
        Take action @action at state @state

        Params
        ------
        state: int
            current state
        action: int
            action taken

        Return
        ------
        (next_state, reward): (int, int)
            a tuple of next state and reward
        '''
        next_state = state + action
        if next_state == 0:
            reward = self.rewards[0]
        elif next_state == self.n_states + 1:
            reward = self.rewards[2]
        else:
            reward = self.rewards[1]
        return next_state, reward
```
To calculate the RMSE, we need to compute the true value of states, which can be achieved with the help of `get_true_value()` function. Here we apply Bellman equations to calculate the true value of states.
```python
def get_true_value(random_walk, gamma):
    '''
    Calculate true value of @random_walk by Bellman equations

    Params
    ------
    random_walk: RandomWalk
    gamma: float
        discount factor
    '''
    P = np.zeros((random_walk.n_states, random_walk.n_states))
    r = np.zeros((random_walk.n_states + 2, ))
    true_value = np.zeros((random_walk.n_states + 2, ))
    
    for state in random_walk.states:
        next_states = []
        rewards = []

        for action in random_walk.actions:
            next_state = state + action
            next_states.append(next_state)

            if next_state == 0:
                reward = random_walk.rewards[0]
            elif next_state == random_walk.n_states + 1:
                reward = random_walk.rewards[2]
            else:
                reward = random_walk.rewards[1]
            rewards.append(reward)

        for state_, reward_ in zip(next_states, rewards):
            if not random_walk.is_terminal(state_):
                P[state - 1, state_ - 1] = random_walk.action_prob * 1
                r[state_] = reward_
        
    u = np.zeros((random_walk.n_states, ))
    u[0] = random_walk.action_prob * 1 * (-1 + gamma * random_walk.rewards[0])
    u[-1] = random_walk.action_prob * 1 * (1 + gamma * random_walk.rewards[2])

    r = r[1:-1]
    true_value[1:-1] = np.linalg.inv(np.identity(random_walk.n_states) - gamma * P).dot(0.5 * (P.dot(r) + u))
    true_value[0] = true_value[-1] = 0

    return true_value
```
In this random walk experiment, we simply use random policy as our action selection.
```python
def random_policy(random_walk):
    '''
    Choose an action randomly

    Params
    ------
    random_walk: RandomWalk
    '''
    return np.random.choice(random_walk.actions)
```
Now it is time to implement our algorithm.
```python
def n_step_temporal_difference(V, n, alpha, gamma, random_walk):
    '''
    n-step TD

    Params
    ------
    V: np.ndarray
        value function
    n: int
        number of steps
    alpha: float
        step size
    random_walk: RandomWalk
    '''
    state = random_walk.start_state
    states = [state]

    T = float('inf')
    t = 0
    rewards = [0] # dummy reward to save the next reward as R_{t+1}

    while True:
        if t < T:
            action = random_policy(random_walk)
            next_state, reward = random_walk.take_action(state, action)
            states.append(next_state)
            rewards.append(reward)
            if random_walk.is_terminal(next_state):
                T = t + 1
        tau = t - n + 1 # updated state's time
        if tau >= 0:
            G = 0 # return
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += np.power(gamma, i - tau - 1) * rewards[i]
            if tau + n < T:
                G += np.power(gamma, n) * V[states[tau + n]]
            if not random_walk.is_terminal(states[tau]):
                V[states[tau]] += alpha * (G - V[states[tau]])
        t += 1
        if tau == T - 1:
            break
        state = next_state
```
As usual, we are going illustrate our result in the main function.
```python
if __name__ == '__main__':
    n_states = 19
    start_state = 10
    gamma = 1
    random_walk = RandomWalk(n_states, start_state)
    true_value = get_true_value(random_walk, gamma)

    episodes = 10
    runs = 100
    ns = np.power(2, np.arange(0, 10))
    alphas = np.arange(0, 1.1, 0.1)

    errors = np.zeros((len(ns), len(alphas)))
    for n_i, n in enumerate(ns):
        for alpha_i, alpha in enumerate(alphas):
            for _ in tqdm(range(runs)):
                V = np.zeros(random_walk.n_states + 2)
                for _ in range(episodes):
                    n_step_temporal_difference(V, n, alpha, gamma, random_walk)
                    rmse = np.sqrt(np.sum(np.power(V - true_value, 2) / random_walk.n_states))
                    errors[n_i, alpha_i] += rmse

    errors /= episodes * runs

    for i in range(0, len(ns)):
        plt.plot(alphas, errors[i, :], label='n = %d' % (ns[i]))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()
    plt.savefig('./random_walk.png')
    plt.close()
```
</div>

This is our result after completing running the code.
<figure>
	<img src="/assets/images/2022-04-08/random_walk.png" alt="Random Walk with n-step TD" style="display: block; margin-left: auto; margin-right: auto; width: 500px"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

### $\boldsymbol{n}$-step TD Control
{: #n-step-td-control}
Similarly, we can apply $n$-step TD methods to control task. In particular, we will combine the idea of $n$-step update with Sarsa, a control method we previously have defined above.

#### $\boldsymbol{n}$-step Sarsa
{: #n-step-sarsa}
As usual, to apply our method to control problem, rather than taking into account states, we instead consider state-action pairs $s,a$ in order to learn the value functions, $\,q_\pi(s,a)$, of them.  
Recall that the target in *one-step Sarsa* update is
\begin{equation}
G_{t:t+1}\doteq R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})
\end{equation}
Similar to what we have done in the previous part of [$n$-step TD Prediction](#n-step-td-prediction), we can redefine the new target of our $n$-step update
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1} R_{t+n}+\gamma^n Q_{t+n-1}(S_{t+n},A_{t+n}),
\end{equation}
for $n\geq 0,0\leq t\<T-n$, with $G_{t:t+n}\doteq G_t$ if $t+n\geq T$. The **$\boldsymbol{n}$-step Sarsa** is then can be defined as:
\begin{equation}
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha\left[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\right],\hspace{1cm}0\leq t\<T,
\end{equation}
while the values of all other state-action pairs remain unchanged: $Q_{t+n}(s,a)=Q_{t+n-1}(s,a)$, for all $s,a$ such that $s\neq S_t$ or $a\neq A_t$.  

From this definition of $n$-step Sarsa, we can easily derive the multiple step version of Expected Sarsa, called **$\boldsymbol{n}$-step Expected Sarsa**.
\begin{equation}
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha\left[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\right],\hspace{1cm}0\leq t\<T,
\end{equation}
where the target of the update is defined as:
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\bar{V}\_{t+n-1}(S_{t+n}),\hspace{1cm}t+n\<T,\tag{8}\label{8}
\end{equation}
with $G_{t:t+n}=G_t$ for $t+n\geq T$, where $\bar{V}\_t(s)$ is the *expected approximate value* of state $s$, using the estimated action value at time $t$, under the target policy $\pi$:
\begin{equation}
\bar{V}\_t(s)\doteq\sum_a\pi(a|s)Q_t(s,a),\hspace{1cm}\forall s\in\mathcal{S}
\end{equation}
If $s$ is terminal, then its expected approximate value is defined to be zero.  

Pseudocode of the $n$-step Sarsa algorithm is given right below.
<figure>
	<img src="/assets/images/2022-04-08/n-step-sarsa.png" alt="n-step Sarsa" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

When taking the value of $n$ from $1$ to $\infty$, similarly, we also obtain a corresponding spectrum ranging from *one-step Sarsa* to *Monte Carlo*.
<figure>
	<img src="/assets/images/2022-04-08/n-step-td-state-action-diagram.png" alt="Backup diagram of n-step TD for state-action values" style="display: block; margin-left: auto; margin-right: auto; width: 570px; height: 370px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: The backup diagram of $n$-step methods for state-action values</figcaption>
</figure>

### Off-policy $\boldsymbol{n}$-step TD
{: #off-policy-n-step-td}
Recall that off-policy methods are ones that learn the value function of a *target policy*, $\,\pi$, while follows a *behavior policy*, $\,b$. In this section, we will be considering an off-policy $n$-step TD, or in specifically, $n$-step TD using **Importance Sampling**[^3].

#### $\boldsymbol{n}$-step TD with Importance Sampling
{: #n-step-td-is}
In $n$-step methods, returns are constructed over $n$ steps, so we are interested in the relative probability of just those $n$ actions. Thus, by weighting updates by *importance sampling ratio*, $\,\rho_{t:t+n-1}$, which is the relative probability under the two policies $\pi$ and $b$ of taking $n$ actions from $A_t$ to $A_{t+n-1}$:
\begin{equation}
\rho_{t:h}\doteq\prod_{k=t}^{\min(h,T-1)}\frac{\pi(A_k|S_k)}{b(A_k|S_k)},
\end{equation}
we can get the **off-policy $\boldsymbol{n}$-step TD** method.
\begin{equation}
V_{t+n}(S_t)\doteq V_{t+n-1}(S_t)+\alpha\rho_{t:t+n-1}\left[G_{t:t+n}-V_{t+n-1}(S_t)\right],\hspace{1cm}0\leq t\<T
\end{equation}
Similarly, we have the **off-policy $\boldsymbol{n}$-step Sarsa** method.
\begin{equation}
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha\rho_{t:t+n-1}\left[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\right],\hspace{0.5cm}0\leq t \<T\tag{9}\label{9}
\end{equation}
The **off-policy $\boldsymbol{n}$-step Expected Sarsa** uses the same update as \eqref{9} except that it uses $\rho_{t+1:t+n-1}$ as its importance sampling ratio instead of $\rho_{t+1:t+n}$ and also has \eqref{8} as its target.  

Here is pseudocode of the off-policy $n$-step Sarsa.
<figure>
	<img src="/assets/images/2022-04-08/off-policy-n-step-sarsa.png" alt="Off-policy n-step Sarsa" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### $\boldsymbol{n}$-step Tree Backup
{: #n-step-tree-backup}

The idea of tree-backup update is to start with the target of the one-step update, which is defined as the first reward plus the discounted estimated value of the next state. This estimated value is computed as the weighted sum of estimated action values. Each weight corresponding to an action is proportional to its probability of occurrence. In particular, the target of one-step tree-backup update is: 
\begin{equation}
G_{t:t+1}\doteq R_{t+1}+\gamma\sum_a\pi(a|S_{t+1})Q_t(S_{t+1},a),\hspace{1cm}t\<T-1
\end{equation}
which is the same as that of Expected Sarsa. With two-step update, for a certain action $A_{t+1}$ taken according to the behavior policy, $\,b$ (i.e.,$\,b(A_{t+1}|S_{t+1})=1$), one step later, the estimated value of the next state similarly now, can be computed as:
\begin{equation}
\pi(A_{t+1}|S_{t+1})\Big(R_{t+2}+\gamma\pi(a|S_{t+2})Q_{t+1}(S_{t+2},a)\Big)
\end{equation}
The target of two-step update, which also is defined as sum of the first reward received plus the discounted estimated value of the next state therefore, can be computed as
\begin{align}
G_{t:t+2}&\doteq R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+1}(S_{t+1},a) \\\\ &\hspace{1cm}+\gamma\pi(A_{t+1}|S_{t+1})\Big(R_{t+2}+\gamma\pi(a|S_{t+2})Q_{t+1}(S_{t+2},a)\Big) \\\\&=R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+1}(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+2},
\end{align}
for $t\<T-2$. Hence, the target of the $n$-step tree-backup update recursively can be defined as:
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma\sum_{a\neq A_{t+1}}\pi(a|S_{t+1})Q_{t+n-1}(S_{t+1},a)+\gamma\pi(A_{t+1}|S_{t+1})G_{t+1:t+n}
\end{equation}
for $t\<T-1,n\geq 2$. The $n$-step tree-backup update can be illustrated through the following diagram
<figure>
	<img src="/assets/images/2022-04-08/3-step-tree-backup.png" alt="3-step tree-backup" style="display: block; margin-left: auto; margin-right: auto; width: 110px; height: 375px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 4</b>: The backup diagram of 3-step tree-backup</figcaption>
</figure>

With this definition of the target, we now can define our **$\boldsymbol{n}$-step tree-backup** method as:
\begin{equation}
Q_{t+n}(S_t,A_t)\doteq Q_{t+n-1}(S_t,A_t)+\alpha\Big[G_{t:t+n}-Q_{t+n-1}(S_t,A_t)\Big],\hspace{1cm}0\leq t\<T
\end{equation}
while the values of all other state-action pairs remain unchanged: $Q_{t+n}(s,a)=Q_{t+n-1}(s,a)$, for all $s,a$ such that $s\neq S_t$ or $a\neq A_t$. Pseudocode of the n-step tree-backup algorithm is given below.
<figure>
	<img src="/assets/images/2022-04-08/n-step-tree-backup.png" alt="n-step tree-backup" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

#### $\boldsymbol{n}$-step $Q(\sigma)$
{: #n-step-q-sigma}


## References
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] <span id='td-convergence'>Sutton, R.S. [Learning to predict by the methods of temporal differences](https://doi.org/10.1007/BF00115009). Mach Learn 3, 9–44 (1988).</span>  

[3] <span id='q-learning-watkins'>Chris Watkins. [Learning from Delayed Rewards](https://www.researchgate.net/publication/33784417_Learning_From_Delayed_Rewards). PhD Thesis (1989)</span>  

[4] Hado Hasselt. [Double Q-learning](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html). NIPS 2010  

[5] Shangtong Zhang. [Reinforcement Learning: An Introduction implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)  

[6] <span id='random_walk'>Singh, S.P., Sutton, R.S. [Reinforcement learning with replacing eligibility traces](https://doi.org/10.1007/BF00114726). Mach Learn 22, 123–158 (1996).</span>  


## Footnotes
[^1]: It is a special case of [n-step TD](#n-step-td) and TD($\lambda$).
[^2]: Bootstrapping is to update estimates  of the value functions of states based on estimates of value functions of other states.
[^3]: For the definition of Importance Sampling method, you can read more in this [section]({% post_url 2021-08-21-monte-carlo-in-rl %}#is).