import torch
import torch.nn.functional as F

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    """Compute cosine distance.
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    x = normalize(x, axis=-1)
    y = normalize(y, axis=-1)
    dist = 1 - torch.mm(x, y.t())
    return dist

class TripletLoss(object):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
        metric (str, optional): which distance metric to use. Default is 'euclidean'.
    """
    
    def __init__(self, margin=0.3, metric='euclidean'):
        self.margin = margin
        if metric == 'euclidean':
            self.dist_func = euclidean_dist
        elif metric == 'cosine':
            self.dist_func = cosine_dist
        else:
            raise ValueError('Invalid metric: {}'.format(metric))

    def __call__(self, feat, pid):
        """
        Args:
            feat (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            pid (torch.Tensor): person IDs (corresponding labels).
        
        Returns:
            torch.Tensor: triplet loss value.
            torch.Tensor: average positive pair distance.
            torch.Tensor: average negative pair distance.
        """
        dist_mat = self.dist_func(feat, feat)
        
        # For each anchor, find the hardest positive and negative
        mask = pid.expand(len(pid), len(pid)).eq(pid.expand(len(pid), len(pid)).t())
        
        dist_ap, dist_an = [], []
        for i in range(len(dist_mat)):
            dist_ap.append(dist_mat[i][mask[i]].max())
            dist_an.append(dist_mat[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        
        # Compute triplet loss
        y = torch.ones_like(dist_an)
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        
        return loss, dist_ap.mean(), dist_an.mean()