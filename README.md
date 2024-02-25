Implementing CNP, NP, and ANP for learning purpose. So far, variants are implemented individually; many of them, however, shares similar architecture, enabling future integration. 

## Model Architecture

<figure>
  <img
  src=https://i.ibb.co/Js1B7RB/model-figure-new-1-page-001.jpg>
  <figcaption>Kim, Hyunjik & Mnih, Andriy & Schwarz, Jonathan & Garnelo, Marta & Eslami, Ali & Rosenbaum, Dan & Vinyals, Oriol & Teh, Yee. (2019). Attentive Neural Processes. </figcaption>
</figure>
### Encoder

encodes observations into corresponding hidden variables. $f_{\theta}$ can be MLP (Conditonal Neural Processes, Neural Processes) or self-attnetion (Attentive Neural Processes). 

$$\begin{equation}r_c = f_{\theta}(\mathcal{D}_c)\end{equation}$$

### Aggregator

aggregates the hidden variables into a global representation in a *permutation-invariant* way, such as 

1. mean = equally weighted, i.e. $\frac{1}{N}\sum{r_c}$ (Condtional Neural Processes, Neural Processes)
2. cross-attention = weighted based on the query, i.e. $\text{cross-attention}(k=x_c, v=r_c, q=x_t)$. (Attentive Neural Processes)

When assuming no latent variable (Conditional Neural Processes, Attentive Neural Processes, Transformer Neural Processes), the aggregation can be written as 

$$\begin{equation}R = h_{\phi}(r_c)\end{equation}$$

When assuming latent variable for modelling functional uncertainty (Neural Processes),

$$\begin{equation}z \sim h_{\phi}(r_c) = \mathcal{N}(\mu(r_c), I\sigma(r_c))\end{equation}$$

by introducing the latent variable, however, the conditonal likelihood becomes intractable. Moreover, the latent modelling can be ignored / the latent distribution is not necessary meaningful when the decoder is powerful.

**We can define both the (deterministic) dependence on context, $R$, and the stochastic process $z$**. 

### Decoder 

predicts target based on the global representation, aka our priors, and the target locations. $g_\psi$ can be MLP or self-attention. 


$$\begin{equation}
    y_t = g_{\psi}(x_t, R)
\end{equation}$$

## Objectives
1. Joint distribution objective with latent variable: $[C|\empty], [T|\empty]$
2. Conditional distribution objective with latent variable: $[T|C]$
3. Conditional distribution objective without latent variable: $\text{det}$

<figure>
  <img
  src="empirical%20evaluation.png">
  <figcaption>Le, Tuan Anh. “Empirical Evaluation of Neural Process Objectives.” (2018).</figcaption>
</figure>

## Reference

```
@misc{garnelo2018conditional,
      title={Conditional Neural Processes}, 
      author={Marta Garnelo and Dan Rosenbaum and Chris J. Maddison and Tiago Ramalho and David Saxton and Murray Shanahan and Yee Whye Teh and Danilo J. Rezende and S. M. Ali Eslami},
      year={2018},
      eprint={1807.01613},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{garnelo2018neural,
      title={Neural Processes}, 
      author={Marta Garnelo and Jonathan Schwarz and Dan Rosenbaum and Fabio Viola and Danilo J. Rezende and S. M. Ali Eslami and Yee Whye Teh},
      year={2018},
      eprint={1807.01622},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@misc{kim2019attentive,
      title={Attentive Neural Processes}, 
      author={Hyunjik Kim and Andriy Mnih and Jonathan Schwarz and Marta Garnelo and Ali Eslami and Dan Rosenbaum and Oriol Vinyals and Yee Whye Teh},
      year={2019},
      eprint={1901.05761},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
```
@inproceedings{Le2018EmpiricalEO,
  title={Empirical Evaluation of Neural Process Objectives},
  author={Tuan Anh Le},
  year={2018},
  url={https://api.semanticscholar.org/CorpusID:89610077}
}
```
