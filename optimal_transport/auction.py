import numpy as np
from numba import jit, prange


@jit(nopython=True)
def cost_matrix(a, b, p=2):
    """
    Numba-based JIT-compiled function to compute the cost matrix between
    two sets of points.

    Parameters
    ----------
    a : ndarray (num_samples, signal_dim)
        Set of points in graph space.
    b : ndarray (num_samples, signal_dim)
        Another set of points in graph space.
    p : int, optional
        The order of the p-norm used to calculate distances between points
        in a and b. Default is 2.

    Returns
    -------
    ndarray (num_samples, num_samples)
        Cost matrix representing the pairwise distances between points in
        sets a and b.

    """

    num_samples = len(a)
    matrix = np.empty((num_samples, num_samples), dtype=np.float32)

    for i in range(num_samples):
        for j in range(num_samples):
            matrix[i, j] = np.linalg.norm(a[i] - b[j], ord=p)**p

    return matrix


@jit(nopython=True, nogil=True, parallel=True)
def auction_algorithm(a, b, p, e_start, e_end, e_fac):
    """
    Numba-based JIT-compiled and parallelised implementation of the auction
    algorithm.

    Bertsekas, D. P. (1988), The auction algorithm: A distributed relaxation
    method for the assignment problem, Annals of operations research 14(1),
    105-123.

    Parameters
    ----------
    a : ndarray (batch_size, num_samples, signal_dim)
        Graph-space transformed data for set 1.
    b : ndarray (batch_size, num_samples, signal_dim)
        Graph-space transformed data for set 2.
    p : int
        The order of the p-norm used to calculate distances between points in
        a and b.
    e_start : float
        Starting minimum positive bidding increment.
    e_end : float
        Final minimum positive bidding increment.
    e_fac : float
        Factor by which the minimum positive bidding increment is reduced each
        e-scaling loop.

    Returns
    -------
    ndarray (batch_size, num_samples)
        Optimal assignments between a and b.

    """

    length = a.shape[1]

    # Allocate memory for final optimal assignments across all sets
    assignments = np.empty((a.shape[0], a.shape[1]), dtype=np.int32)

    # Save dimensions of complete unassigned array
    init_unassigned = np.arange(0, length, dtype=np.int32)

    # Parallelize loop over all samples in the batch with 'prange'
    for sample in prange(a.shape[0]):

        c_matrix = cost_matrix(a[sample], b[sample], p=p)
        # Define object prices outside of e-scaling loops
        object_prices = np.zeros(length, dtype=np.float32)

        e = e_start  # Set minimum positive bidding increment to start value
        while e > e_end:

            # Allocate memory for assignments array for current sets
            # a[sample] and b[sample]
            sample_assignments = np.empty(length, dtype=np.int32)
            # If object is unassigned store -1
            sample_assignments[:] = -1

            # All bidders are unassigned to begin with and stored as
            # indices in unassigned
            unassigned = init_unassigned.copy()
            num_unassigned = len(unassigned)

            while num_unassigned > 0:

                # --> Bidding phase <--
                # Randomly select a single unassigned bidder
                random_int = np.random.randint(0, num_unassigned)
                bidder = unassigned[unassigned >= 0][random_int]

                # Calculate total value of each object for selected bidder
                values = -c_matrix[bidder] - object_prices

                J = 0  # Initial index of best value
                V = values[J]  # Initial best value
                W = values[J]  # Initial second best value

                for object in range(len(values)):
                    object_value = values[object]

                    if object_value > V:
                        W = V  # Downgrade previous best value
                        V = object_value  # Update best object value
                        J = object  # Update index of the best value

                    elif object_value > W:
                        W = object_value  # Update second best object value

                B = V - W + e  # Calculate bidding increment for best object

                # --> Assignment phase <--
                # Raise price of best object by bidding increment
                object_prices[J] += B

                # Check if object J is already assigned to a bidder
                if sample_assignments[J] >= 0:
                    # Return previously assigned bidder to unassigned array
                    unassigned[sample_assignments[J]] = sample_assignments[J]
                    num_unassigned += 1  # Increment unassigned bidders

                sample_assignments[J] = bidder  # Assign bidder to best object
                unassigned[bidder] = -1  # Remove bidder from unassigned array
                num_unassigned -= 1  # Deduct 1 from unassigned bidders

            e /= e_fac  # Divide minimum positive bidding increment by e_fac

        assignments[sample] = sample_assignments

    return assignments
