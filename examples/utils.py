import numpy as np
import matplotlib.pyplot as plt


def cloud_plot(a, b=None, assignments=None, **kwargs):
    """
    Create a scatter plot to visualize point clouds and potential assignments.

    Parameters
    ----------
    a : ndarray
        Array representing the first set of points.
    b : ndarray, optional
        Array representing the second set of points. If provided, both sets
        will be plotted.
    assignments : ndarray, optional
        Array representing the indices of assignments between points in sets
        'a' and 'b'.
    **kwargs : additional keyword arguments
        s : int, optional
            Marker size for the scatter plot. Default is 80.
        linewidths : float, optional
            Linewidth of markers in the scatter plot. Default is 1.5.
        alpha : float, optional
            Opacity of markers in the scatter plot. Default is 0.8.
        size : tuple, optional
            Figure size in inches. Default is (5, 5).
        t_plan : ndarray, optional
            Transportation permutation matrix representing the bijection
            between set 'a' and 'b'. If provided, it will be used to visualize
            the assignments.

    Returns
    -------

    """

    s = kwargs.pop('s', 80)
    linewidths = kwargs.pop('linewidths', 1.5)
    alpha = kwargs.pop('alpha', 0.8)
    size = kwargs.pop('size', (5, 5))
    t_plan = kwargs.pop('t_plan', None)

    plt.figure(figsize=size)
    plt.axis('off')
    plt.scatter(a[:, 0], a[:, 1],
                c='b',
                s=s,
                edgecolors='k',
                linewidths=linewidths,
                alpha=alpha)

    if b is not None:
        plt.scatter(b[:, 0], b[:, 1],
                    color='r',
                    s=s,
                    edgecolors='k',
                    linewidths=linewidths,
                    alpha=alpha)

    if assignments is not None:
        plt.plot(np.stack((a[assignments, 0], b[:, 0])),
                 np.stack((a[assignments, 1], b[:, 1])),
                 color='k',
                 linewidth=1)
    elif t_plan is not None:
        i_ind, j_ind = np.nonzero(t_plan.value > 1e-5)
        for k in range(len(i_ind)):
            plt.plot(np.stack((a[i_ind[k], 0], b[j_ind[k], 0])),
                     np.stack((a[i_ind[k], 1], b[j_ind[k], 1])),
                     color='k',
                     linewidth=1)
    plt.show()
