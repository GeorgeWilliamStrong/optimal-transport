# Optimal Transport for Signal Analysis

PyTorch implementation of the transportation $L_{p}$ distance ($TL_{p}$) for signal analysis with an implementation of the auction algorithm for efficiently solving the Monge optimal assignment problem.

In contrast to the p-Wasserstein metric which is used to compare probability distributions (e.g. non-negative normalised data), the $TL_{p}$ distance can be used more generally to compare *any* signals of interest. This includes signed and unnormalised data such as images and audio.

The transportation $L_{p}$ distance is defined as follows:

```math
TL_{p}(\alpha, \beta) = \underset{\gamma}{\textup{min}}\int_{X}\int_{Y}\gamma(x,y)c_{\lambda, p}(x, y, \alpha, \beta)\textup{d}x\textup{d}y
```
where
```math
c_{\lambda,p}(x,y,\alpha,\beta)=\frac{1}{\lambda}\left|x-y\right|_{p}^{p}+\left|\alpha(x)-\beta(y)\right|_{p}^{p},
```
subject to the following constraints: 
```math
\int_{Y}\gamma(x, y) \textup{d}y=\alpha(x),
```
```math
\int_{X}\gamma(x,y)\textup{d}x=\beta(y),
```
```math
\gamma(x, y)\geq0.
```

FIX MATH DEFINITION - USE MORE GENERAL FORM FROM PAPER.

TLP DISTANCE IS EQUIVALENT TO P-WASSERSTEIN DISTANCE OF THE GRAPH OF ALPHA AND BETA. CONSIDERING A NUMERICAL IMPLEMENTATION, THE GRAPH OF A DISCRETE ONE-DIMENSIONAL SIGNAL SUCH AS TIME SERIES DATA, WOULD BE A CLOUD OF EQUAL MASS POINTS DEFINED WITHIN THE TIME AND AMPLITUDE DIMENSION. ASSUMING EACH MEASURE (ALPHA AND BETA) CONTAIN THE SAME NUMBER OF EQUAL MASS POINTS, BIRKHOFF'S THEOREM STATES that the optimal solution simplifies to a permutation matrix, where every entry is either a 1 or 0. Finding the optimal permutations amounts to solving a linear sum assignment problem.

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
