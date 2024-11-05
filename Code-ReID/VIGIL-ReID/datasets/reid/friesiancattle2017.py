import glob
import re
import os.path as osp
from ..base_dataset import ReIDDataset, Datum  # Add Datum import here
from ..build_dataset import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class FriesianCattle2017(ReIDDataset):
    """FriesianCattle2017.

    Reference:
        Andrew et al. Visual Localisation and Individual Identification of 
        Holstein Friesian Cattle via Deep Learning. ICCV 2017.

    Dataset statistics:
        - identities: 66 (train) + 18 (gallery) + 18 (query)
        - images: 752 (train) + 97 (gallery) + 85 (query)
    """
    dataset_dir = 'FriesianCattle2017'

    def __init__(self, cfg, verbose=True):
        root = cfg.DATASETS.ROOT_DIR
        super(FriesianCattle2017, self).__init__(root, verbose)

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

    def _process_train(self, dir_path):
        """Process training data."""
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'\d+_-?\d+_\d+_?.*')
        
        print(f"\nProcessing training directory: {dir_path}")
        print(f"Found {len(img_paths)} images")
        
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).group().split("_")[0:2])
            pid_container.add(pid)
            
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        print("PID mapping:")
        for pid, label in pid2label.items():
            print(f"Original PID: {pid} -> New label: {label}")

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).group().split("_")[0:2])
            pid = int(pid2label[pid])
            camid = int(camid)
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))

        return data

    def _process_test(self, dir_path):
        """Process test data."""
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'\d+_-?\d+_\d+_?.*')
        
        print(f"\nProcessing test directory: {dir_path}")
        print(f"Found {len(img_paths)} images")
        
        for img_path in sorted(img_paths):
            print(f"\nProcessing: {osp.basename(img_path)}")
            pid, camid = map(int, pattern.search(img_path).group().split("_")[0:2])
            print(f"  PID: {pid}, CamID: {camid}")
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))
        
        pids = [item.pid for item in data]
        camids = [item.camid for item in data]
        print(f"\nStatistics for {osp.basename(dir_path)}:")
        print(f"  Total images: {len(data)}")
        print(f"  Unique PIDs: {sorted(set(pids))}")
        print(f"  Unique Camera IDs: {sorted(set(camids))}")
        
        from collections import Counter
        pid_counts = Counter(pids)
        print("\nImages per PID:")
        for pid in sorted(pid_counts.keys()):
            print(f"  PID {pid}: {pid_counts[pid]} images")
            
        return data