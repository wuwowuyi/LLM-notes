GRPO is introduced in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), and is a variant of [PPO](https://arxiv.org/abs/1707.06347).

🤔 The GRPO algorithm looks to me is like a clipped + KL regularized version of the basic reinforce policy gradient algorithm, especially the outcome supervision method. 
The process supervision replaces $r(\tau)$ as a reward-to-go value, where $\tau$ is the trajectory. 
$r(\tau)$ has high variance, that's why a group of outputs for each question $q$ is sampled, I believe. So GRPO trades sampling efficiency for computation efficiency (less memory footprint).

The DeepSeekMath paper also talks about [RFT (Rejection Sampling Fine-tuning)](https://arxiv.org/abs/2308.01825). This RFT method first samples data from $\pi_{sft}$ or $\pi_\theta$ (the optimizing policy), and then filter wrong and duplicated answers to generate new training data to 
further finetune $\pi_\theta$. <br> 
🤔 It makes sense that online RFT with data sampled from $\pi_\theta$ has better performance, since sampling from $\pi_{sft}$ has a distributional shift.

### PPO
PPO is a policy gradient algorithm. The objective function of PPO is:
$\displaystyle J_{PPO}(\theta)=E_{q \sim P, o \sim \pi_{\theta_{old}}}\left[\min\left(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}A_t, \text{clip}(\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1-\varepsilon, 1+\varepsilon)A_t\right)\right]$

where $q$ is a question (prompt), and $o$ is the response sampled from policy $\pi_{\theta_{old}}$ given $q$.

Note $\pi_{\theta_{old}}$ is just the optimizing policy from last iteration, not $\pi_{sft}$. 

Because the sampling policy is $\pi_{\theta_{old}}$ rather than $\pi_\theta$, importance sampling <br>
$\displaystyle \text{ratio}\_t(\theta)=\frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}$ <br>
is used.

Further, in order to stabilize training, preventing too large gradient steps, this probability ratio $\text{ratio}_t(\theta)$ is clipped to lie in the interval $[1-\varepsilon, 1+\varepsilon]$ where $\varepsilon$ is a hyperparameter, like 0.2.

#### Generalized advantage estimator 
$A_t$ is the advantage computed according to the [GAE algorithm](https://arxiv.org/abs/1506.02438), and is an exponentially-weighted average of $k$-step advantage estimators.

$k$-step advantage estimator with a discount factor $\gamma$ is as follows:<br>
one step $A^{(1)}\_t=\delta_t = r_t + \gamma V(s_{t+1})-V(s_t)$,<br>
two step $A^{(2)}\_t=\delta_t + \gamma \delta_{t+1} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})-V(s_t)$,<br>
$k$-step $\displaystyle A^{(k)}\_t=\sum^{k-1}\_{l=0}\gamma^l\delta_{t+l} = r_t + \gamma r_{t+1} + ... + \gamma^{k-1}r_{t+k-1}+\gamma^k V(s_{t+k})-V(s_t)$<br>
where $V(s)$ is the state value function.

When using PPO to RL finetuning LLMs, the reward $r_t$ of an output $o$ consists two parts, a per-token step KL penalty, and a score computed by the reward model at the last step for the entire output sequence $o$. 

Since $V(s)$ is just an estimation, one step advantage estimator $A^{(1)}_t$ is biased. Because there is only one step, it has small variance. As $k \to \infty$, bias is reduced but variance increased. To tradeoff bias and variance, generalized advantage $A_t$ is an exponentially weighted average of all $k$-step estimators,

 $\displaystyle A_t =(1-\lambda)\left(A^{(1)}\_{t} + \lambda A^{(3)}\_t + \lambda^2A^{(3)}\_t + ...\right) = \sum^\infty_{l=0}(\gamma\lambda)^l\delta^V_{t+l}$<br>
 
where hyperparameter $\lambda \in [0, 1]$ controls the bias variance tradeoff. 

When $\lambda = 0$, $A_t = A^{(1)}_t$ has high bias and low variance.<br>
when $\lambda = 1$, we recover Monte-Carlo sampling which has no bias but high variance.

### GRPO
In PPO, to compute $A_t$ we need fit a state function $V(s)$ alongside with policy $\pi_\theta$. Training two large transformers at the same time is very resource intensive.

GRPO proposes to use the average reward of multiple sampled outputs rather than fitting a state value function. Specifically, for each question $q$, instead of sampling a single output, GRPO samples a group of outputs $\\{o_1, o_2, ...,o_G\\}$, and then optimizes the policy model by the follow objective:

$\displaystyle J_{GRPO}(\theta)=E_{q \sim P, o_i \sim \pi_{\theta_{old}}}\left[\frac{1}{G}\sum^G_{i=1}\left(\min\left(\text{ratio}\_{i, t}\hat{A}\_{i, t}, \text{clip}(\text{ratio}\_{i, t},1-\varepsilon,1+\varepsilon)\hat{A}\_{i, t}\right)-\beta D_{KL}[\pi_\theta||\pi_{ref}]\right)\right]$

where<br>
$\displaystyle \text{ratio}\_{i, t} = \frac{\pi_\theta(o_{i,t}|q, o_{i<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i<t})}$.

Compared with PPO, the only difference is how $\hat{A}\_{i, t}$ is computed. GRPO proposes two methods to estimate $\hat{A}_{i, t}$, outcome supervision and process supervision.

#### Outcome supervision
A reward model gives a score of each output $\\{o_1, o_2, ...,o_G\\}$ **at the last step** (i.e., for the entire output), yielding $r = \\{r_1, r_2, ...,r_G\\}$, then set the reward of **all steps** as $\hat{A}_{i, t}=\tilde{r}_i = \frac{r_i - mean(r)}{std(r)}$ where $mean(r)$ is group mean and $std(r)$ group standard deviation.

#### Process supervision
Process supervision requires to first train a process reward model which can give a score for each step, which means a per-step preference dataset is needed to train the reward model. (🤔 this is possible with math questions, but not with all types of questions.)

Given a question $q$ and its group outputs $\\{o_1, o_2, ...,o_G\\}$, the process reward model can generate scores $R=\\{\\{r_1^{index(1)},...,r_1^{index(K_1)}\\},...,\\{r_G^{index(1)},...,r_G^{index(K_G)}\\}\\}$, where $index(j)$ is the last token of the $j$-step, and $K_i$ is the total number of steps in the $o_i$ output. 

Let $\tilde{r}\_i^{index(j)} = \frac{r_i^{index(j)} - mean(R)}{std(R)}$, <br>
we have $\hat{A}\_{i, t}=\sum_{index(j) \ge t}\tilde{r}_i^{index(j)}$, which is a reward-to-go value, the sum of all rewards as of the current step.

#### KL penalty
The per-token KL penalty is not included in the reward as PPO does. In stead, it is estimated with the following unbiased estimator ([Schulman, 2020](http://joschu.net/blog/kl-approx.html)):

$\displaystyle D_{KL}[\pi_\theta||\pi_{ref}] = \frac{\pi_{ref}(o_{i,t}|q, o_{i<t})}{\pi_\theta(o_{i,t}|q, o_{i<t})} - \log\frac{\pi_{ref}(o_{i,t}|q, o_{i<t})}{\pi_\theta(o_{i,t}|q, o_{i<t})} - 1$

which is guaranteed to be positive.

#### Iterative training
Iterative training works in a similar way as the online training in the RLHF methods.
At the end of each iteration, the reward model is **continuously trained** based on sampling results from $\pi_\theta$, with 10% historical data. (🤔 How are the samples labelled? )
And in the next iteration, the reference model $\pi_{ref}$ is replaced with the policy model $\pi_\theta$ trained in the last iteration.


