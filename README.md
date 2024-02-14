# Optimal Transport for Signal Analysis

PyTorch implementation of the transportation $L_{p}$ distance ($TL_{p}$) for signal analysis and machine learning [[1]](https://arxiv.org/abs/1609.08669), with an implementation of the auction algorithm [[2]](https://link.springer.com/article/10.1007/BF02186476) for efficiently solving the Monge optimal assignment problem.

This implementation is general, and no assumption is made about the dimensionality of the input signals. The `TransportLpLoss()` criterion can be equivalently applied to 1D time series data, 3D image data or point cloud data (where the graph transform is not required).

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

## Formulation

In contrast to conventional optimal transport distances used to compare probability distributions (e.g. non-negative normalised data), the $TL_{p}$ distance can be used more generally to provide a global comparison between *any* signals of interest. This includes signed and unnormalised data such as images and audio.

The $TL_{p}$ distance between two signals, $\alpha$ and $\beta$, is defined as follows:

```math
TL_{p}(\alpha, \beta) = W_{p}(\hat{\alpha}, \hat{\beta}) = \left (  \underset{\gamma\in\prod(\hat{\alpha}, \hat{\beta})}{\textup{min}}\int_{\Omega \times \Omega } || x - y ||_{p}^{p} \textup{d}\gamma(x, y) \right )^{\frac{1}{p}}
```

where $W_{p}$ represents the $p$-Wasserstein metric and ($\hat{\alpha}, \hat{\beta}$) represent the graphs of the input signals. The minimum is taken over probability measures $\gamma$ on $\Omega \times \Omega$ subject to the following constraints:

```math
\int_{\Omega}\gamma(x, y) \textup{d}y=\hat{\alpha}(x);

\int_{\Omega}\gamma(x, y) \textup{d}x=\hat{\beta}(y);

\gamma(x, y)\geq 0.
```

The coupling or transportation plan, $\gamma(x, y)$, represents the amount of mass or energy to be moved between $\hat{\alpha}(x)$ and $\hat{\beta}(y)$, such that $|| x - y ||_{p}^{p}$ is minimised over $\Omega \times \Omega$. 

From a numerical perspective it is advantageous to consider the discrete version of this problem:

```math
TL_{p}(\alpha, \beta) = \left ( \underset{T}{\textup{min}}\sum_{i=1}^{n}\sum_{j=1}^{m}\gamma_{i, j}|| x_{i} - y_{j} ||_{p}^{p} \right )^{\frac{1}{p}},
```

subject to the constraints:

```math
\sum_{j=1}^{m}\gamma_{i, j} = \hat{\alpha}_{i};

\sum_{i=1}^{n}\gamma_{i, j} = \hat{\beta}_{j};

\gamma_{i,j}\geq 0.
```

Under this formulation, the graph of a discrete one-dimensional signal such as a time series corresponds to a two-dimensional cloud of unit-mass points defined within the time and amplitude dimension. Assuming each measure being compared ($\hat{\alpha}$ and $\hat{\beta}$) have the same number of equal-mass points, Birkhoff's theorem [[3]](https://cir.nii.ac.jp/crid/1570572699525842816) states that the optimal solution for $\gamma$ simplifies to a permutation matrix $\gamma^{\sigma}$, where every entry is either a 1 or 0. This can be further simplified by using $\sigma$ to represent a bijective function between $\alpha$ and $\beta$, and finding optimal permutations then amounts to solving the following linear sum assignment problem:

```math
TL_{p}(\alpha, \beta) = \left ( \underset{\sigma}{\textup{min}}\frac{1}{n}\sum_{i=1}^{n}||\hat{\alpha}_{i}-\hat{\beta}_{\sigma(i)}||_{p}^{p} \right ) ^{\frac{1}{p}}
```

subject to the constraints:

```math
\sum_{j=1}^{n}\gamma^{\sigma}_{i, j} = 1;

\sum_{i=1}^{n}\gamma^{\sigma}_{i, j} = 1;

\gamma^{\sigma}_{i,j}\geq 0.
```

Efficient numerical algorithms exist for solving such problems, such as the auction algorithm [[2]](https://link.springer.com/article/10.1007/BF02186476) used in this implementation.

## References

[[**1**]](https://arxiv.org/abs/1609.08669) Thorpe, M., Park, S., Kolouri, S., Rohde, G. K. & Slepcev, D. (2017), ‘A transportation $L_{p}$ distance for signal analysis’, *Journal of mathematical imaging and vision* **59**(2), 187–210.

[[**2**]](https://link.springer.com/article/10.1007/BF02186476) Bertsekas, D. P. (1988), ‘The auction algorithm: A distributed relaxation method for the assignment problem’, *Annals of operations research* **14**(1), 105–123.

[[**3**]](https://cir.nii.ac.jp/crid/1570572699525842816) Birkhoff, G. (1946), ‘Tres observaciones sobre el algebra lineal’, Univ. Nac. Tucuman, Ser. A 5, 147–154.

