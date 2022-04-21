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
- [$n$-step TD](#n-step-td)
	- [$n$-step TD Prediction](#n-step-td-prediction)
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

<button type="button" class="collapsible" id="codeP">Click to show the code</button>
<div class="codePanel" id="codePdata" markdown="1">
<br>
We begin by importing necessary packages we will be using

```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
```
Our first step is to define the environment, gridworld with a cliff, which is created by height, width, cliff region, start state, goal state, actions and rewards.
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
As we have seen so far in Sarsa and Q-learning, the action-value function, $Q$, has been over estimated because 
- In Q-learning, the target policy is the greedy policy given the current action values, which is defined with a maximization operation.
- In Sarsa, the policy is often $\varepsilon$-greedy, which also involves a maximization operation.
- And under these methods, a maximization over estimated values is used implicitly as an estimate of the maximum value.  

That overestimation can lead to a significant positive bias, which is called *maximization bias*.

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

[5] Shangtong Zhang. [Reinforcement Learning: An Introduction implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)  


## Footnotes
[^1]: It is a special case of [n-step TD](#n-step-td) and TD($\lambda$).
[^2]: Another popular variant of Q-learning is [Deep Q-learning](https://www.nature.com/articles/nature14236), which was introduced by Deepmind in 2015. We're gonna talk about it in the post of Function approximation.