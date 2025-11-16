# aggregators.py
"""Robust aggregation methods for Byzantine-resilient federated learning"""

import numpy as np


def nnm_preprocessing(gradients, n_workers, f_byzantine):
    """Nearest Neighbor Mixing: average each gradient with its n-f nearest neighbors"""
    gradients = [np.asarray(g) for g in gradients]
    ref_shape = gradients[0].shape
    
    grads = []
    for g in gradients:
        if g.shape != ref_shape or not np.all(np.isfinite(g)):
            grads.append(np.zeros(ref_shape))
        else:
            grads.append(g)
    
    mixed_gradients = []
    
    for i in range(n_workers):
        distances = []
        for j in range(n_workers):
            dist = np.linalg.norm(grads[i] - grads[j])
            distances.append((dist, j))
        
        distances.sort(key=lambda x: x[0])
        nearest_indices = [idx for _, idx in distances[:n_workers - f_byzantine]]
        
        nearest_grads = np.stack([grads[j] for j in nearest_indices])
        mixed_grad = np.mean(nearest_grads, axis=0)
        mixed_gradients.append(mixed_grad)
    
    return mixed_gradients


def _smoothed_weiszfeld(points, T=8, nu=0.1):
    """Compute approximate geometric median via smoothed Weiszfeld iterations"""
    pts = [p for p in points if np.all(np.isfinite(p))]
    if len(pts) == 0:
        return np.zeros_like(points[0])
    
    pts = [np.asarray(p, dtype=float) for p in pts]
    z = np.zeros_like(pts[0])
    m = len(pts)
    alphas = np.full(m, 1.0 / m)
    
    for _ in range(T):
        betas = []
        for k in range(m):
            dist = np.linalg.norm(z - pts[k])
            betas.append(alphas[k] / max(dist, nu))
        beta_sum = sum(betas)
        if beta_sum > 0:
            z = sum(w * b for w, b in zip(pts, betas)) / beta_sum
    
    return z


def rfa_aggregation(gradients, T=8, nu=0.1):
    """RFA: geometric median via smoothed Weiszfeld"""
    return _smoothed_weiszfeld(gradients, T=T, nu=nu)


def krum_aggregation(gradients, f_byzantine):
    """KRUM: select gradient with smallest sum of distances to n-f-1 nearest neighbors"""
    n = len(gradients)
    scores = []
    
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances.append(dist)
        
        distances.sort()
        score = np.sum(distances[:n - f_byzantine - 1])
        scores.append(score)
    
    best_idx = np.argmin(scores)
    return gradients[best_idx]


def coordinate_median_aggregation(gradients):
    """Coordinate-wise median aggregation"""
    valid_grads = [g for g in gradients if np.all(np.isfinite(g))]
    if len(valid_grads) == 0:
        return np.zeros_like(gradients[0])
    
    arr = np.stack(valid_grads)
    return np.median(arr, axis=0)


def get_aggregator(aggregator_type):
    """Factory function to get the appropriate aggregator"""
    aggregators = {
        'rfa': lambda grads, f_byz: rfa_aggregation(grads),
        'krum': lambda grads, f_byz: krum_aggregation(grads, f_byz),
        'cm': lambda grads, f_byz: coordinate_median_aggregation(grads),
        'median': lambda grads, f_byz: coordinate_median_aggregation(grads)
    }
    
    if aggregator_type.lower() not in aggregators:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")
    
    return aggregators[aggregator_type.lower()]