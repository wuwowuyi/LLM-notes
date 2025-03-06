
## Distilling the Knowledge in a Neural Network
[paper link](https://arxiv.org/abs/1503.02531)

(🤔See a trained model as a function that maps input $x$ to output $y$, not just trained parameters): We tend to identify the knowledge in a trained model with the learned parameter values and this makes it hard to see how we can change the form of the model but keep the same knowledge. A more abstract view of the knowledge, that frees it from any particular instantiation, is that it is **a learned mapping from input vectors to output vectors**.

(🤔Relative probabilities of incorrect answers help generalization) The normal training objective is to maximize the average log probability of the correct answer, but a side-effect of learning is that the trained model assigns probabilities to all of the incorrect answers and even when these probabilities are very small, some of them are much larger than others. **The relative probabilities of incorrect answer tells us a lot about how the cumbersome model tends to generalize**. 

### Distillation
Typically, the output distribution $q_i$, computed from raw logits $z_i$, where $T$ is a temperature:

$\displaystyle q_i = \frac{exp(z_i/T)}{\sum_jexp(z_j/T)}$

In the simplest form, knowledge is transferred by training the small model on a transfer dataset and using the soft target distribution produced by the cumbersome model with a high temperature $T$ in its softmax. (🤔 increase temperature to learn the relative probabilities of incorrect answers, or in other words, **similarity structure**.)

When the true labels are known for all or some of the transfer set, we can use **a weighted average of two objective function**:
* first is the cross entropy with the soft targets, using a higher temperature $T > 1$. (🤔 Is kl divergence better in this case? from the description above, kl divergence helps to learn the similarity structure and as a result better generalization.)
* second is the cross entropy with the true label, using $T = 1$

We found the best results were generally obtained by using **a considerably low weight on the second objective function**.

Note: **the magnitudes of the gradients produced by the soft targets scale as $\frac{1}{T^2}$, it is important to multiply them by $T^2$ _when using both hard and soft targets_**. This ensures that the relative contribution of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters.
(🤔 In other words, we should multiply the weight on the first objective by $T^2$.)

Write some code to help understand.
```python
from torch import nn
from torch.nn import functional as F

def loss(logits, lm_logits, labels, temperature, alpha):
    """
    logits: small model output logits
    lm_logits: large model output logits
    temperature: temperature
    alpha: weight of the cross entropy with the true label, like 0.1?
    """
    p = F.log_softmax(logits/temperature, dim=1)
    q = F.softmax(lm_logits/temperature, dim=1)
    loss1 = nn.KLDivLoss(reduction="batchmean")(p, q)
    loss2 = F.cross_entorpy(logits, labels)
    loss = alpha * loss2 + ((1 -  alpha) * temperature ** 2) * loss1
    return loss
```








