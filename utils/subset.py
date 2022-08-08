import numpy as np
import utils.coreset as coreset


def get_coreset(gradient_est, 
                labels, 
                N, 
                B, 
                num_classes, 
                equal_num=True,
                optimizer="LazyGreedy",
                metric='euclidean'):
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p) 
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool

    Returns 
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = coreset.get_orders_and_weights(
            B, 
            gradient_est, 
            metric, 
            y=labels, 
            equal_num=equal_num, 
            num_classes=num_classes,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')

    return subset, subset_weights, ordering_time, similarity_time, cluster


def get_random_subset(B, N):
    print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]

    return subset


