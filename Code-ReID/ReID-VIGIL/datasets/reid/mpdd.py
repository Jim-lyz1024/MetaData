import glob
import re
import os.path as osp
from ..base_dataset import ReIDDataset, Datum
from ..build_dataset import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class MPDD(ReIDDataset):
    """
    MPDD: Multi-Pose Dog Dataset
    Reference:
     He et al. Animal Re-identification Algorithm for Posture Diversity. ICASSP 2023.
    
    Dataset statistics:
    # identities: 95 (train) + 96 (test)
    # images: 1032 (train) + 521 (gallery) + 104 (query)
    """
    dataset_dir = 'MPDD'
    
    def __init__(self, cfg, verbose=True):
        root = cfg.DATASETS.ROOT_DIR
        super(MPDD, self).__init__(root, verbose)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _extract_pid_camid(self, img_path):
        """Extract person ID and camera ID from file name."""
        img_name = osp.basename(img_path)  # e.g., 0_c1s1_6_7.jpg
        split_name = img_name.split('_')
        pid = int(split_name[0])
        
        # Extract camera ID from format like 'c1s1'
        cam_str = split_name[1]  # e.g., 'c1s1'
        # Extract number after 'c'
        camid = int(cam_str[1])
        
        return pid, camid

    def _process_train(self, dir_path):
        """Process training data."""
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        print(f"\nProcessing training directory: {dir_path}")
        print(f"Found {len(img_paths)} images")
        
        # First pass: collect all PIDs
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = self._extract_pid_camid(img_path)
            pid_container.add(pid)
        
        # Create mapping from PIDs to labels
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        print("PID mapping:")
        for pid, label in pid2label.items():
            print(f"Original PID: {pid} -> New label: {label}")
        
        # Second pass: create data list
        for img_path in sorted(img_paths):
            pid, camid = self._extract_pid_camid(img_path)
            pid = int(pid2label[pid])  # Map to new label
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))
        
        return data

    def _process_test(self, dir_path):
        """Process test data."""
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        print(f"\nProcessing test directory: {dir_path}")
        print(f"Found {len(img_paths)} images")
        
        # Process each image
        for img_path in sorted(img_paths):
            print(f"\nProcessing: {osp.basename(img_path)}")
            pid, camid = self._extract_pid_camid(img_path)
            print(f"  PID: {pid}, CamID: {camid}")
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))
        
        # Print statistics
        pids = [item.pid for item in data]
        camids = [item.camid for item in data]
        print(f"\nStatistics for {osp.basename(dir_path)}:")
        print(f"  Total images: {len(data)}")
        print(f"  Unique PIDs: {sorted(set(pids))}")
        print(f"  Unique Camera IDs: {sorted(set(camids))}")
        
        # Print per-PID statistics
        from collections import Counter
        pid_counts = Counter(pids)
        print("\nImages per PID:")
        for pid in sorted(pid_counts.keys()):
            print(f"  PID {pid}: {pid_counts[pid]} images")
        
        return data