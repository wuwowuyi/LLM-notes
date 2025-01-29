My understanding on paper [Statistical rejection sampling improves preference optimization](https://arxiv.org/abs/2309.06657).

Two important reference works, [DPO](https://arxiv.org/abs/2305.18290) and [SLiC](https://arxiv.org/abs/2210.00045).
### Sampling distributional shift

During RL finetuning, typically a KL divergence constraint is used to prevent the fine-tuned model from drifting too far from the pretrained/SFT-ed model.

It can be proved that the optimal distribution $\pi^* = \text{argmax}_{\pi}E_{y \sim \pi(y|x)}[r(x, y)]$  such that $D_{KL}[\pi(y|x)||\pi_{sft}(y|x)] < \epsilon$ is:
 $\displaystyle \pi^* = \frac{1}{Z(x)}\pi_{sft}(y|x) \cdot\exp(\frac{1}{\gamma}r(x, y))$
where:
- $D_{KL}(\pi||\pi_{sft})$ is the KL-divergence between $\pi$ and $\pi_{sft}$
- $\gamma$ is like a "temperature" parameter. When $\gamma \to 0$ it means full reward exploitation and $\gamma \to \infty$ means $\pi^* = \pi_{sft}$ with full exploration.  
- $Z(x)=E_{y \sim \pi_{sft}}[\exp(\frac{1}{\gamma}r(x,y))]$, i.e., $Z(x)$ is a normalizing constant wrt. $y \sim \pi_{sft}$. 

Note in the optimization objective $E_{y \sim \pi(y|x)}[r(x, y)]$, $y \sim \pi(y|x)$, i.e., given prompt $x$, the response $y$ is sampled from the optimizing policy $\pi$, not $\pi_{sft}$ or any other policies $\pi_{unk}$, where $\pi_{unk}$ denotes any other policies. Existing works like DPO do not train from prompt response pairs sampled from $\pi$, which results a distributional shift. 

This paper propose a training framework to alleviate the distributional shift to achieve better performance. Specifically: 
* Fit a pairwise ranking reward model $r_{\psi}(x, y)$ from a human preference dataset $D_{hf}$ collected from any policy
* By using statistical rejection sampling, we can approximate generating samples from the optimal policy $\pi^*$ by sampling from the proposal distribution $\pi_{sft}$
* Label accepted samples $\mathcal{Y}$ using trained reward model $r_{\psi}(x, y)$
* train a policy model on $\mathcal{Y}$ in the same way as DPO

The authors found that the language model **learns better from an explicit reward model** because comparing between two responses (reward) is easier to learn than generating high quality responses (policy).

:question: My Question: The proposed training method alleviates the distributional shift in training policy model. But it looks to me the reward model still has a similar problem since it is trained on $D_{hf}$. Maybe because the reward model is easier to train, the distributional shift has less negative effect?
### Statistical rejection sampling
Statistical rejection sampling is key to approximate generating samples from the optimal policy.

The estimation of $\pi^*$ can be viewed as a density estimation problem.
In this case $\pi_{sft}$ is used as the proposal distribution from which we can generate samples.
Steps given in the paper follow the general statistical rejection sampling, and Appendix A.1 gives the Python code.

The key part is to compute $\displaystyle \frac{\pi_{r_{\psi}}(y|x)}{M\cdot\pi_{sft}(y|x)}$.
We know in statistical rejection sampling we must have $M\cdot\pi_{sft}(y|x) \ge \pi_{r_{\psi}}(y|x)$. Usually we use the smallest M, ie., $\displaystyle M = \max\frac{\pi_{r_{\psi}}(y|x)}{\pi_{sft}(y|x)}$
Since the optimal policy $\displaystyle \pi_{r_{\psi}}(y|x) = \frac{1}{Z_{\psi}(x)}\pi_{sft}(y|x) \cdot\exp(\frac{1}{\beta}r_{\psi}(x, y))$, we have
$\displaystyle \frac{\pi_{r_{\psi}}(y|x)}{\pi_{sft}(y|x)} = \frac{1}{Z_{\psi}(x)}\cdot\exp(\frac{1}{\beta}r_{\psi}(x, y))$ --- (1)
then $\displaystyle M= \frac{1}{Z_{\psi}(x)}\max[\exp(\frac{1}{\beta}r_{\psi}(x, y))] = \frac{1}{Z_{\psi}(x)}\exp[\frac{1}{\beta}\max(r_{\psi}(x, y))]$ --- (2)
Put (1) and (2) together, $\displaystyle \frac{\pi_{r_{\psi}}(y|x)}{M\cdot\pi_{sft}(y|x)} = \exp[\frac{1}{\beta}(r_{\psi}(x, y) - \max r_{\psi}(x, y))]$
$r_{\psi}(x, y)$ is given by the trained reward model, and $\max(r_{\psi}(x, y))$ is approximated as max reward of a mini-batch of samples. 

Section 5.2 in the paper gives details on generating preference pairs:
* For a prompt $x$, sample 64 responses from SFT model, followed by subsampling 8 responses by statistical rejection sampling (Algorithm 1 in appendix A.1).
* Two approaches are given in generating preference pairs from the $n$ accepted responses. ($n=8$ in this case)
	* "first-round-rank": construct $n/2$ pairs and get them labeled. 
	* "tournament rank": pick a single winner and form $n-1$ pairs. 

The paper points out that the tournament ranking used in SLiC **introduces bias towards higher reward sequences**.

The paper says, statistical rejection sampling is better than best-of-N or top-k-over-N algorithms which has the issue of **reward hacking** because it trusts the reward model without regularization. In other words, statistical rejection sampling **makes a better tradeoff between reward exploitation and exploration**, while the best-of-N or top-k-over-N means pure exploitation and as a result more vulnerable to reward hacking.
### Loss function

Based on the loss function in DPO and SLiC, this paper proposes a new loss function:
$\displaystyle L_{\text{hinge-norm}}(\pi_{\theta}|\pi_{sft}, D_p) = \mathbb{E}_{(x,y_w,y_l) \sim D_p}[\max(0, 1-[\gamma\log\frac{\pi_{\theta}(y_w|x)}{\pi_{sft}(y_w|x)} - \gamma\log\frac{\pi_{\theta}(y_l|x)}{\pi_{sft}(y_l|x)}])]$