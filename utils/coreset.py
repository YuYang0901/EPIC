import time
import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction


def faciliy_location_order(c, X, y, metric, num_per_class, weights=None, optimizer="LazyGreedy"):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), data=X, metric=metric, mode='dense')
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer=optimizer,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)
    cluster = -np.ones(N)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        cluster[i] = np.argmax(S[i, order])
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    cluster[cluster>=0] += c * num_per_class

    return class_indices[order], sz, greedy_time, S_time, cluster


def get_orders_and_weights(B, X, metric, y=None, weights=None, equal_num=False, num_classes=10, optimizer="LazyGreedy"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        total = np.sum(num_per_class)
        diff = total - B
        chosen = set()
        for i in range(diff):
            j = np.random.randint(C)
            while j in chosen or num_per_class[j] <= 0:
                j = np.random.randint(C)
            num_per_class[j] -= 1
            chosen.add(j)

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times, cluster_all = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], weights, optimizer=optimizer), classes))

    order_mg = np.concatenate(order_mg_all).astype(np.int32)
    weights_mg = np.concatenate(cluster_sizes_all).astype(np.float32)
    class_indices = [np.where(y == c)[0] for c in classes]
    class_indices = np.concatenate(class_indices).astype(np.int32)
    class_indices = np.argsort(class_indices)
    cluster_mg = np.concatenate(cluster_all).astype(np.int32)[class_indices]
    assert len(order_mg) == len(weights_mg)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, cluster_mg
    return vals



