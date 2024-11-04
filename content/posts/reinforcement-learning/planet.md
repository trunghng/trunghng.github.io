---
title: "PlaNet"
date: 2024-10-13T17:12:41+07:00
tags: [deep-reinforcement-learning, model-based, my-rl]
draft: true
math: true
eqn-number: true
---
> A model-based RL method that learns the dynamics model from pixels and chooses actions through online planning in a compact latent space.
<!--more-->

### Problem setup
We consider a discrete-time partially observable Markov decision process. At each time-step $t$, we have state $s_t$, image observation $o_t$, continuous action vector $a_t$, and scalar reward $r_t$ following the stochastic dynamics, which consists of four components
\begin{align}
&\small\text{Transition function:}&&s_t\sim\text{p}(s_t\vert s_{t-1},a_{t-1})\nonumber \\\\ &\small\text{Observation function:}&&o_t\sim\text{p}(o_t\vert s_t)\nonumber \\\\ &\small\text{Reward function:}&&r_t\sim\text{p}(r_t\vert s_t)\nonumber \\\\ &\small\text{Policy:}&&a_t\sim\text{p}(a_t\vert o_{\leq t},a_{\lt t})\nonumber,
\end{align}
where we assume a fixed initial state $s_0$ without loss of generality. Our goal is to find a policy $\text{p}(a_t\vert o_{\leq t},a_{<t})$ that maximizes the expected sum of rewards $\mathbb{E}\_\text{p}\left[\sum_{t=1}^{H}r_t\right]$ taken over the distributions of the environment and the policy.

### Recurrent state-space model{#rssm}
For planning, PlaNet use a **recurrent state-space model (RSSM)** that can make predictions in latent space.

<figure>
	<img src="/images/planet/latent-dynamics-model-designs.png" alt="latent-dynamics"/>
	<figcaption style='text-align: center;'><b>Figure 1</b>: (taken from <a href='#planet-paper'>PlaNet paper</a>) <b>Latent dynamics model designs.</b></figcaption>
</figure>

#### Latent dynamics model
Say we consider sequences $\\{o_t,a_t,r_t\\}\_{t=1}^H$ with discrete time-step $t$, image observations $o_t$, continuous action vectors $a_t$, and scalar rewards $r_t$. Typically, a latent state-space model shown in Figure 1b represents the structure of a POMDP. It defines the generative process of the observations and rewards using a hidden state sequence $\\{s_t\\}\_{t=1}^H$
\begin{align}
&\small\text{Transition model:}&&s_t\sim p(s_t\vert s_{t-1},a_{t-1})\nonumber \\\\ &\small\text{Observation model:}&&o_t\sim p(o_t\vert s_t)\nonumber \\\\ &\small\text{Reward model:}&&r_t\sim p(r_t\vert s_t)\nonumber,
\end{align}
where we assume a fixed initial state $s_0$ without loss of generality. Specifically, in PlaNet:
<ul class='number-list'>
	<li>
		The transition model is Gaussian with mean and variance parameterized by a feed-forward network.
	</li>
	<li>
		The observation model is Gaussian with mean parameterized by a deconvolutional network and identity covariance.
	</li>
	<li>
		The reward model is a scalar Gaussian with mean parameterized by a feed-forward network and unit variance.
	</li>
</ul>

#### Variational encoder
Since the model is non-linear, and thus is intractable, the state posteriors $p(s_{1:H}\vert a_{1:H})$ that are required for parameter learning cannot be computed directly. We instead use an encoder
\begin{equation}
q(s_{1:H}\vert o_{1:H},a_{1:H})=\prod_{t=1}^{H}q(s_t\vert s_{t-1},a_{t-1},o_t)
\end{equation}
to infer approximate state posteriors from past observations and actions, where $q(s_t\vert s_{t-1},a_{t-1},o_t)$ is a diagonal Gaussian with mean and variance parameterized by a convolutional network followed by a feed-forward network.

Using the encoder, we construct a variational bound on the data log-likelihood.

### Planning in latent space


### Preferences
[1] <span id='planet-paper'>Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. [Learning Latent Dynamics for Planning from Pixels]()</span>.

