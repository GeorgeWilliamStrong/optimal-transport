# Optimal Transport for Signal Analysis

PyTorch implementation of the transportation $L_{p}$ distance ($TL_{p}$) for signal analysis [1] with an implementation of the auction algorithm [2] for efficiently solving the Monge optimal assignment problem.

## Formulation

In contrast to conventional optimal transport distances used to compare probability distributions (e.g. non-negative normalised data), the $TL_{p}$ distance can be used more generally to provide a global comparison between *any* signals of interest. This includes signed and unnormalised data such as images and audio.

The $TL_{p}$ distance between two signals, $\alpha$ and $\beta$, is defined as follows:

```math
TL_{p}(\alpha, \beta) = W_{p}(\hat{\alpha}, \hat{\beta}) = \left (  \underset{\gamma\in\prod(\hat{\alpha}, \hat{\beta})}{\textup{min}}\int_{\Omega \times \Omega } || x - y ||_{p}^{p} \textup{d}\gamma(x, y) \right )^{\frac{1}{p}}
```

where $W_{p}$ represents the $p$-Wasserstein metric and ($\hat{\alpha}, \hat{\beta}$) represent the graphs of the input signals. The minimum is taken over probability measures $\gamma$ on $\Omega \times \Omega$ subject to the following constraints:

```math
\int_{\Omega}\gamma(x, y) \textup{d}y=\hat{\alpha}(x);
```
```math
\int_{\Omega}\gamma(x, y) \textup{d}x=\hat{\beta}(y);
```
```math
\gamma(x, y)\geq 0.
```

The coupling or transportation plan, $\gamma(x, y)$, represents the amount of mass or energy to be moved between $\hat{\alpha}(x)$ and $\hat{\beta}(y)$, such that $|| x - y ||_{p}^{p}$ is minimised over $\Omega \times \Omega$. 

From a numerical perspective it is advantageous to consider the discrete version of this problem:

```math
TL_{p}(\alpha, \beta) = \left ( \underset{T}{\textup{min}}\sum_{i=1}^{n}\sum_{j=1}^{m}\gamma_{i, j}|| x_{i} - y_{j} ||_{p}^{p} \right )^{\frac{1}{p}}
```



CONSIDERING A NUMERICAL IMPLEMENTATION, THE GRAPH OF A DISCRETE ONE-DIMENSIONAL SIGNAL SUCH AS TIME SERIES DATA, WOULD BE A CLOUD OF EQUAL MASS POINTS DEFINED WITHIN THE TIME AND AMPLITUDE DIMENSION. ASSUMING EACH MEASURE (ALPHA AND BETA) CONTAIN THE SAME NUMBER OF EQUAL MASS POINTS, BIRKHOFF'S THEOREM STATES that the optimal solution simplifies to a permutation matrix, where every entry is either a 1 or 0. Finding the optimal permutations amounts to solving a linear sum assignment problem.

As the transformed predicted and observed data traces, $\mathcal{T}(G(\mathbf{m})_{s,r})$ and $\mathcal{T}(\mathbf{d}_{s,r})$, contain the same number of equal-mass points, Birkhoff's theorem \citep{birkhoff1946tres} states that the optimal solution for Kantorovich's problem \eqref{eqn:Kantorovich's problem} simplifies to a permutation matrix, $T_{\sigma}$, where every entry in $T_{\sigma}$ is either a 1 or 0 \citep{peyre2019computational}. $T_{\sigma}$ represents a bijection between points $\mathcal{T}(G(\mathbf{m})_{s,r})$ and $\mathcal{T}(\mathbf{d}_{s,r})$, and finding the optimal permutations $\sigma$ amounts to solving a linear sum assignment problem

## Quickstart

```sh
git clone https://github.com/GeorgeWilliamStrong/optimal-transport
cd optimal-transport
pip install -e .
```

## Usage

The $TL_{p}$ distance has been implemented as a PyTorch module. It can be used to provide a global OT-based comparison between n-dimensional signals.

```python
from optimal_transport import TransportLpLoss

criterion = TransportLpLoss()
loss = criterion(input, target)
```

During `forward()`, clones of the `input` and `target` variables are detached from the computation graph. Their graph-space transform is taken, and the optimal assignment problem between the resultant sets is solved. The optimal assignments are then used to formulate the transportation $L_{p}$ distance between the `input` and `target` variables that exist *within the computation graph* such that the module is differentiable through `loss.backward()`.

## References

[[**1**]](https://arxiv.org/abs/1609.08669) Thorpe, M., Park, S., Kolouri, S., Rohde, G. K. & Slepcev, D. (2017), ‘A transportation $L_{p}$ distance for signal analysis’, *Journal of mathematical imaging and vision* **59**(2), 187–210.

[[**2**]](https://link.springer.com/article/10.1007/BF02186476) Bertsekas, D. P. (1988), ‘The auction algorithm: A distributed relaxation method for the assignment problem’, *Annals of operations research* **14**(1), 105–123.
