import numpy as np
import torch
from collections import defaultdict

def compute_distance_matrix(query_features, gallery_features, metric='euclidean'):
    """Compute distance matrix between query and gallery features.
    
    Args:
        query_features (torch.Tensor): Query features with shape (num_query, feat_dim)
        gallery_features (torch.Tensor): Gallery features with shape (num_gallery, feat_dim)
        metric (str): Distance metric, ['euclidean', 'cosine']
    
    Returns:
        torch.Tensor: Distance matrix with shape (num_query, num_gallery)
    """
    if metric == 'euclidean':
        return pairwise_distance(query_features, gallery_features)
    elif metric == 'cosine':
        return cosine_distance(query_features, gallery_features)
    else:
        raise ValueError(f'Unknown distance metric: {metric}')

def pairwise_distance(x, y):
    """Compute euclidean distance between two tensors.
    
    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor
    
    Returns:
        torch.Tensor: Distance matrix
    """
    m, n = x.size(0), y.size(0)
    x_pow = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
    y_pow = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat = x_pow + y_pow
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
    
    # Handle numerical stability
    dist_mat = dist_mat.clamp(min=1e-12).sqrt()
    
    return dist_mat

def cosine_distance(x, y):
    """Compute cosine distance between two tensors.
    
    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor
    
    Returns:
        torch.Tensor: Distance matrix
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    
    # Cosine similarity to distance
    dist_mat = 1 - torch.mm(x_norm, y_norm.t())
    
    return dist_mat

def compute_ranked_results(distmat, query_pids, gallery_pids, topk=100):
    """Compute ranking results.
    
    Args:
        distmat (torch.Tensor): Distance matrix
        query_pids (torch.Tensor): Query person IDs
        gallery_pids (torch.Tensor): Gallery person IDs
        topk (int): Top-k results to return
        
    Returns:
        list: List of ranked indices for each query
        list: List of ranking results containing matched IDs for each query
    """
    num_q, num_g = distmat.size()
    
    if num_g < topk:
        topk = num_g
    
    # Get top-k indices for each query
    indices = torch.argsort(distmat, dim=1)[:, :topk]
    
    # Get matched IDs for each query
    matches = gallery_pids[indices] == query_pids.view(-1, 1)
    
    return indices.cpu(), matches.cpu()

def compute_ap_cmc(matches, return_all=False):
    """Compute Average Precision (AP) and CMC curves.
    
    Args:
        matches (torch.Tensor): Binary match indicators
        return_all (bool): Whether to return scores at all ranks
        
    Returns:
        tuple:
            - float: Average Precision score
            - torch.Tensor: CMC scores at different ranks
    """
    # Convert to numpy for computation
    matches = matches.numpy()
    
    # Compute AP
    relevants = matches.sum(1)
    if relevants.sum() == 0:
        return 0, np.zeros(matches.shape[1])
        
    aps = []
    for match_row, n_relevant in zip(matches, relevants):
        if n_relevant == 0:
            continue
            
        pos_idx = np.where(match_row)[0]
        pos_ranks = pos_idx + 1.0
        
        ap = np.sum(np.arange(1, len(pos_ranks) + 1) / pos_ranks) / n_relevant
        aps.append(ap)
    
    # Compute CMC
    cmc = matches.any(axis=1)
    cmc = np.cumsum(cmc) / cmc.shape[0]
    
    if return_all:
        return np.mean(aps), cmc
    else:
        return np.mean(aps), cmc[0] # return AP and rank-1

def evaluate_rank(
    distmat, 
    q_pids, 
    g_pids, 
    q_camids, 
    g_camids, 
    max_rank=50
):
    """Evaluate ranking results.
    
    Args:
        distmat (numpy.ndarray): Distance matrix
        q_pids (numpy.ndarray): Query person IDs
        g_pids (numpy.ndarray): Gallery person IDs
        q_camids (numpy.ndarray): Query camera IDs
        g_camids (numpy.ndarray): Gallery camera IDs
        max_rank (int): Maximum rank to compute
        
    Returns:
        tuple:
            - numpy.ndarray: CMC scores
            - float: mAP score
            - numpy.ndarray: All AP scores
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
        
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # Compute AP for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        # Get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Compute CMC
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # This query has no valid matches
            continue
            
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        
        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        
    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP, all_AP