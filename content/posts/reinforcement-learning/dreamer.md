---
title: "Dreamer"
date: 2024-02-29T14:08:53+07:00
tags: [deep-reinforcement-learning, model-based, my-rl]
math: true
draft: true
eqn-number: true
hideSummary: true
---

### Problem setup
Dreamer works in the scope of a partially observable Markov decision process (POMDP). A partially observable Markov decision process is a tuple of $(\mathcal{S},\mathcal{A},T,R,\Omega ,O,\gamma)$ where
- $(\mathcal{S},\mathcal{A},T,R,\gamma)$ describes a Markov decision process;
- $\Omega$ is a finite set of observations;
- $O:\mathcal{S}\times\mathcal{A}\to\Pi(\Omega)$ is the observation function, which gives, for each action and resulting state, a probability over possible observations, i.e. $O(s',a,o)=P(o\vert s',a)$.

### Dreamer
Dreamer consists of three components, performed in parallel or interleaved:
<ul class='number-list'>
	<li>
		<b>Dynamics learning</b>: learning the latent dynamics model from past experience to predict future rewards from actions and past observations.
	</li>
	<li>
		<b>Behavior learning</b>: learning action and value models from predicted latent trajectories.
	</li>
	<li>
		<b>Environment acting</b>: generating experience by executing the learned action model in the world.
	</li>
</ul>

<figure>
	<img src="/images/dreamer/dreamer.png" alt="dreamer"/>
	<figcaption style='text-align: center;'><b>Figure 1</b>: (taken from <a href='#dreamer-paper'>Dreamer paper</a>) <b>Components of Dreamer</b>. (a) Learn latent dynamics from experience; (b) Learn behavior in imagination; (c) Act in the world</figcaption>
</figure>

#### Behavior learning by latent imagination
Since the compact model states $s_t$ are Markovian, the latent dynamics then define an MDP that is fully observed. Letting $\tau$ denote the time index for all quantities in this MDP, each imagined trajectory $\\{(s_\tau,a_\tau,r_\tau)\\}\_{\tau=t}$ starts at a true state, $s_\tau=s_t$ for $\tau=0$ and follow predictions of:
\begin{align}
&\small\text{Transition model:}&&s_\tau\sim q(s_\tau\vert s_{\tau-1},a_{\tau-1})\nonumber \\\\ &\small\text{Reward model:}&&r_\tau\sim q(r_\tau\vert s_\tau)\nonumber \\\\ &\small\text{Policy:}&&a_\tau\sim q(a_\tau\vert s_\tau)\nonumber
\end{align}
The object is to maximize the expected cumulative imagined rewards $\mathbb{E}\_q\left[\sum_{\tau=t}^{\infty}\gamma^{\tau-t}r_\tau\right]$ taken over the policy $q(a_\tau\vert s_\tau)$.

Consider imagined trajectories with a finite horizon $H$. Within the latent space, Dreamer learns an action model (or policy) $q_\phi$, parameterized by $\phi$, that aims to predict actions that solve the imagination environment
\begin{equation}
a_\tau\sim q_\phi(a_\tau\vert s_\tau)
\end{equation}
At the same time, it also learns a value model $v_\psi$, parameterized by $\psi$, that estimates the expected cumulative imagined rewards that the action model achieves from each state $s_\tau$
\begin{equation}
v_\psi(s_\tau)\approx\mathbb{E}\_{q(\cdot\vert s_\tau)}\left[\sum_{\tau=t}^{t+H}\gamma^{\tau-t}r_\tau\right]
\end{equation}
In the paper, both models are implemented as MLPs. The action model $q_\phi$ outputs a $\tanh$-transformed Gaussian with sufficient statistics predicted by the network, as in [**SAC**]({{<ref"maxent-sql-sac#action-sample">}}).
\begin{equation}
a_\tau=\tanh(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\epsilon),\hspace{1cm}\epsilon\sim\mathcal{N}(0,I)
\end{equation}
In order to learn the action and value models, we need to estimate the state values of imagined trajectories $\\{s_\tau,a_\tau,r_\tau\\}\_{\tau=t}^{t+H}$, Dreamer, in particular, uses the [$\lambda$-return]({{<ref"eligible-traces#lambda-return">}})[^1]:
\begin{align}
V_N^k(s_\tau)&\doteq\mathbb{E}\_{q_\phi,q_\theta}\left[\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\right],\hspace{1cm}h=\min(\tau+k,t+H) \\\\ V_\lambda(s_\tau)&\doteq(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}V_N^n(s_\tau)+\lambda^{H-1}V_N^H(s_\tau),
\end{align}
where $V_N^k$ is the $k$-step return, which estimates rewards beyond $k$ steps with the learned value model $v_\psi(s_h)$.

Once the $\lambda$-returns $V_\lambda(s_\tau)$ for all $s_\tau$ along the imagined trajectories are computed, the parameters $\phi$ and $\psi$ can be updated iteratively by SGD according to the action loss and value loss respectively:
\begin{align}
\mathcal{L}\_\phi&=-\mathbb{E}\_{q_\theta,q_\phi}\left[\sum_{\tau=t}^{t+H}V_\lambda(s_\tau)\right] \\\\ \mathcal{L}\_\psi&=\mathbb{E}\_{q_\theta,q_\phi}\left[\sum_{\tau=t}^{t+H}\frac{1}{2}\Big\Vert v_\psi(s_\tau)-V_\lambda(s_\tau)\Big\Vert^2\right]
\end{align}

#### Latent dynamics learning
There are three approaches for learning representations that are used with Dreamer: reward prediction, image reconstruction and contrastive estimation.

##### Image Reconstruction
We can apply the world model used by [PlaNet]({{<ref"planet#latent-dynamics">}}) to learn the latent dynamics by reconstructing images. Recall that the model consists of four components
\begin{align}
&\small\text{Representation model:}&& p_\theta(s_t\vert s_{t-1},a_{t-1},o_t)\nonumber \\\\ &\small\text{Observation model:}&& q_\theta(o_t\vert s_t)\nonumber \\\\ &\small\text{Reward model:}&& q_\theta(r_t\vert s_t)\nonumber \\\\ &\small\text{Transition model:}&& q_\theta(s_t\vert s_{t-1},a_{t-1})\nonumber
\end{align}


##### Contrastive estimation

### DreamerV2

### DreamerV3

### References
[1] <span id='dreamer-paper'>Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). arXiv preprint, arXiv:1912.01603, 2019</span>.

[2] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193). arXiv preprint, arXiv:2010.02193, 2020.

[3] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). arXiv preprint, arXiv:2301.04104, 2023.

[4] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

### Footnotes
[^1]: $q_\theta$ is the transition model, will be defined in the next section.
