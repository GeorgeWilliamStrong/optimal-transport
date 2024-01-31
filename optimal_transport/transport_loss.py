import numpy as np
import torch

from .auction import auction_algorithm


__all__ = ['TransportLpLoss']


class TransportLpLoss(torch.nn.Module):
    """
    PyTorch module for computing the Transport-Lp loss between two sets of
    points in graph space.

    Thorpe, M., Park, S., Kolouri, S., Rohde, G. K. & Slepcev, D. (2017), A
    transportation lp distance for signal analysis, Journal of mathematical
    imaging and vision 59(2), 187-210.

    Parameters
    ----------
    p : int, optional
        The order of the p-norm used to calculate distances. Default is 2.
    method : Literal['auction', 'scipy'], optional
        The method used to compute the Transport-Lp loss.
    e_start : float, optional
        Starting minimum positive bidding increment for the auction algorithm.
        Default is 1e3.
    e_end : float, optional
        Final minimum positive bidding increment for the auction algorithm.
        Default is 1e-4.
    e_fac : float, optional
        Factor by which the minimum positive bidding increment is reduced in
        each e-scaling loop. Default is 10.

    """

    def __init__(self, p=2, method='auction', **kwargs):
        super(TransportLpLoss, self).__init__()
        self.p = p
        self.method = method
        self.e_start = kwargs.pop('e_start', 1e3)
        self.e_end = kwargs.pop('e_end', 1e-4)
        self.e_fac = kwargs.pop('e_fac', 10)

    def forward(self, input, target):
        """
        Compute the Transport-Lp loss between input and target.

        Parameters
        ----------
        input : torch.Tensor
            Input set of points.
        target : torch.Tensor
            Target set of points.

        Returns
        -------
        torch.Tensor
            Transport-Lp loss between input and target.

        """

        assert input.shape == target.shape, "Shapes are not equal"
        ndim = input.ndim
        if ndim == 1:
            input = input[None]
            target = target[None]

        g_input = self.graph_space_transform(
            input.detach().cpu().clone().numpy())
        g_target = self.graph_space_transform(
            target.detach().cpu().clone().numpy())

        shape = g_input.shape
        num_samples = shape[0]

        if ndim > 2:
            g_input = g_input.reshape(
                (num_samples, np.prod(shape[1:-1]), shape[-1]))
            g_target = g_target.reshape(
                (num_samples, np.prod(shape[1:-1]), shape[-1]))

        if self.method == 'auction':
            assignments = auction_algorithm(g_target,
                                            g_input,
                                            self.p,
                                            self.e_start,
                                            self.e_end,
                                            self.e_fac)
        elif self.method == 'scipy':
            raise NotImplementedError

        if ndim > 2:
            assignments = assignments.reshape(shape[:-1])

        loss = 0
        for sample in range(num_samples):
            for dim in range(ndim - 1):
                loss += torch.norm(g_input[sample, ..., dim] -
                                   assignments[sample],
                                   self.p) ** self.p
            loss += torch.norm(input[sample] -
                               target[sample][assignments[sample]],
                               self.p) ** self.p

        return loss

    def graph_space_transform(self, a):
        """
        Transform the input array a into graph space.

        Parameters
        ----------
        a : ndarray
            Input array.

        Returns
        -------
        ndarray
            Transformed array in graph space.

        """

        dim = a.ndim - 1
        shape = a.shape

        b = np.empty(list(shape) + [dim + 1], dtype=np.float32)
        seq = np.arange(0, shape[-1], 1)

        if dim > 1:
            mesh = np.meshgrid(*[seq] * dim, indexing='ij')
            for i in range(dim):
                b[..., i] = mesh[i]
        elif dim == 1:
            b[..., 0] = seq

        b[..., dim] = a

        return b
