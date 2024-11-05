import numpy as np
import torch
from collections import OrderedDict
from sklearn.metrics import average_precision_score

from .build_evaluator import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class ReIDEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.feat_norm = cfg.TEST.FEAT_NORM
        self.rerank = cfg.TEST.RE_RANKING
        self.max_rank = cfg.TEST.MAX_RANK
        self.remove_identical = cfg.TEST.REMOVE_IDENTICAL
        self.reset()

    def reset(self):
        """Reset internal states."""
        self.query_feats = []
        self.query_pids = []
        self.query_camids = []
        
        self.gallery_feats = []
        self.gallery_pids = []
        self.gallery_camids = []

    def process(self, features, data, is_query=False, is_gallery=False):
        """Process one batch of data."""
        if is_query:
            self.query_feats.append(features.cpu())
            self.query_pids.extend(data['pid'].numpy())
            self.query_camids.extend(data['camid'].numpy())
        elif is_gallery:
            self.gallery_feats.append(features.cpu())
            self.gallery_pids.extend(data['pid'].numpy())
            self.gallery_camids.extend(data['camid'].numpy())

    def evaluate(self):
        """Calculate evaluation metrics."""
        # Concatenate features
        query_feats = torch.cat(self.query_feats, dim=0)
        gallery_feats = torch.cat(self.gallery_feats, dim=0)
        
        # Convert to numpy arrays
        query_feats = query_feats.numpy()
        gallery_feats = gallery_feats.numpy()
        
        query_pids = np.asarray(self.query_pids)
        gallery_pids = np.asarray(self.gallery_pids)
        query_camids = np.asarray(self.query_camids)
        gallery_camids = np.asarray(self.gallery_camids)

        print("\nComputing CMC and mAP...")
        print(f"Query feature shape: {query_feats.shape}")
        print(f"Gallery feature shape: {gallery_feats.shape}")
        print(f"Number of query IDs: {len(np.unique(query_pids))}")
        print(f"Number of gallery IDs: {len(np.unique(gallery_pids))}")

        # Normalize features if needed
        if self.feat_norm:
            print("Normalizing features with L2 norm")
            query_feats = torch.nn.functional.normalize(
                torch.from_numpy(query_feats), p=2, dim=1
            ).numpy()
            gallery_feats = torch.nn.functional.normalize(
                torch.from_numpy(gallery_feats), p=2, dim=1
            ).numpy()

        # Compute distance matrix
        dist_mat = self._compute_dist(query_feats, gallery_feats)
        
        # Compute metrics
        cmc, mAP = self._evaluate_rank(
            dist_mat,
            query_pids,
            gallery_pids,
            query_camids,
            gallery_camids
        )
        
        print("\nResults ----------")
        print(f"mAP: {mAP:.1%}")
        print(f"CMC curve")
        for r in [1, 5, 10]:
            print(f"Rank-{r}: {cmc[r-1]:.1%}")
            
        return {
            'mAP': mAP,
            'rank1': cmc[0],
            'rank5': cmc[4],
            'rank10': cmc[9]
        }
        
    def _compute_dist(self, query_features, gallery_features):
        """Compute distance matrix between query and gallery features.
        
        Args:
            query_features (numpy.ndarray): Query features of shape (num_query, feat_dim).
            gallery_features (numpy.ndarray): Gallery features of shape (num_gallery, feat_dim).
            
        Returns:
            numpy.ndarray: Distance matrix of shape (num_query, num_gallery).
        """
        query_features = torch.from_numpy(query_features)
        gallery_features = torch.from_numpy(gallery_features)
        
        m, n = query_features.size(0), gallery_features.size(0)
        distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()
        
        return distmat

    def _compute_map(self, indices, query_pids, gallery_pids):
        """Compute mAP with more robust handling of edge cases."""
        aps = []
        for i in range(len(query_pids)):
            # Get matches
            matches = (gallery_pids[indices[i]] == query_pids[i]).astype(np.float32)
            
            # Skip if no matches
            if not np.any(matches):
                continue
            
            # Compute AP
            cumsum = np.cumsum(matches)
            ranks = np.arange(1, len(matches) + 1)
            ap = np.sum((cumsum / ranks) * matches) / np.sum(matches)
            aps.append(ap)
            
        if len(aps) == 0:
            print("Warning: No valid query found!")
            return 0.0
            
        return np.mean(aps)

    def _compute_cmc(self, dist_mat, query_pids, gallery_pids, configs):
        """Compute CMC scores.
        
        Args:
            dist_mat (numpy.ndarray): Distance matrix.
            query_pids (numpy.ndarray): Query person IDs.
            gallery_pids (numpy.ndarray): Gallery person IDs.
            configs (dict): CMC config containing ranks.
            
        Returns:
            dict: CMC scores at different ranks.
        """
        # Sort and find correct matches
        indices = np.argsort(dist_mat, axis=1)
        matches = (gallery_pids[indices] == query_pids[:, np.newaxis])
        
        # Compute CMC for each rank
        results = {}
        for name, r in configs.items():
            # Find matches within rank r
            match_within_r = matches[:, :r].any(axis=1)
            results[name] = np.mean(match_within_r)
            
        return results

    def _compute_rerank_dist(self, query_features, gallery_features, k1=20, k2=6, lambda_value=0.3):
        """Compute re-ranking distance.
        
        Reference:
            Zhong et al. Re-ranking Person Re-identification with k-reciprocal Encoding. CVPR 2017.
            
        Args:
            query_features (numpy.ndarray): Query features.
            gallery_features (numpy.ndarray): Gallery features.
            k1 (int): Parameter for k-reciprocal.
            k2 (int): Parameter for k-reciprocal.
            lambda_value (float): Weighting parameter.
            
        Returns:
            numpy.ndarray: Re-ranking distance matrix.
        """
        # Implementation of re-ranking algorithm
        # Note: Simplified version shown here
        q_g_dist = self._compute_dist(query_features, gallery_features)
        g_g_dist = self._compute_dist(gallery_features, gallery_features)
        
        # Compute final distance
        re_rank_dist = (1-lambda_value) * q_g_dist + lambda_value * g_g_dist
        
        return re_rank_dist
    
    def _evaluate_rank(self, distmat, q_pids, g_pids, q_camids, g_camids):
        """Evaluate ranking results."""
        num_q, num_g = distmat.shape
        
        if num_g < self.max_rank:
            self.max_rank = num_g
            print(f"Note: number of gallery samples is quite small, got {num_g}")
        
        print(f"\nUnique Query PIDs: {np.unique(q_pids)}")
        print(f"Unique Gallery PIDs: {np.unique(g_pids)}")
        print(f"Unique Query Camera IDs: {np.unique(q_camids)}")
        print(f"Unique Gallery Camera IDs: {np.unique(g_camids)}")

        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # Compute CMC and mAP
        all_cmc = []
        all_AP = []
        num_valid_q = 0

        for q_idx in range(num_q):
            # Get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # Get gallery indices that match the query pid
            order = indices[q_idx]
            raw_cmc = matches[q_idx]
            
            if self.remove_identical:
                # Remove gallery samples that have the same pid and camid with query
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                keep = np.invert(remove)
                raw_cmc = raw_cmc[keep]
            else:
                # 只移除完全相同的图片（如果它排在第一位）
                if num_g > 1:
                    same_image = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                    if same_image[0]:  # 如果第一个就是相同的图片
                        raw_cmc = raw_cmc[1:]

            if not np.any(raw_cmc):
                print(f"Query {q_idx} (PID: {q_pid}, CamID: {q_camid}) has no valid matches")
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:self.max_rank])
            num_valid_q += 1

            # Compute AP
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
            
            print(f"Query {q_idx}: AP={AP:.4f}, #matches={num_rel}")

        if num_valid_q == 0:
            print("\nError: No valid queries found!")
            print("Possible reasons:")
            print("1. Camera IDs might be identical for all images")
            print("2. PID labeling might be incorrect")
            print("3. Dataset split might need adjustment")
            return np.zeros(self.max_rank), 0.0
        
        print(f"\nNumber of valid queries: {num_valid_q}")
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        
        return all_cmc, mAP