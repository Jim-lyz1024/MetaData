# datasets/reid/stoat.py
import glob
import re
import os.path as osp
from ..base_dataset import ReIDDataset, Datum
from ..build_dataset import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Stoat(ReIDDataset):
    """
    New Zealand (Waiheke Island and South Island) Stoat Dataset
    
    Dataset statistics:
    # train - South Island, gallery - Waiheke Island, query - Waiheke Island  
    # identities: 56 (train) + 5 (gallery) + 5 (query)
    # images: 183 (train) + 13 (gallery) + 13 (query)
    """
    dataset_dir = 'Stoat'
    
    def __init__(self, cfg, verbose=True):
        root = cfg.DATASETS.ROOT_DIR
        super(Stoat, self).__init__(root, verbose)

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
        return self._process_dir(dir_path, is_train=True)

    def _process_test(self, dir_path): 
        return self._process_dir(dir_path, is_train=False)

    def _process_dir(self, dir_path, is_train=True):
        """Process directory.
        Args:
            dir_path (str): directory path
            is_train (bool): training set or test set
        """
        data = []
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        special_cameras = {
            "CREK": 1000, 
            "FC01": 1001,
            "FC11": 1002,
            "GC34": 1003,
            "P164": 1004
        }
        pattern = re.compile(r'\d+_[0-9a-zA-Z]+_\d+')

        # First pass: collect all PIDs
        pid_container = set()
        for img_path in sorted(img_paths):
            pid, camid, _ = pattern.search(img_path).group().split("_")
            pid = int(pid)
            pid_container.add(pid)

        # Create mapping from PIDs to labels for training set
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Second pass: create dataset
        for img_path in sorted(img_paths):
            pid, camid, _ = pattern.search(img_path).group().split("_")
            pid = int(pid)
            if camid in special_cameras:
                camid = special_cameras[camid]
            camid = int(camid)

            # Verify PID range
            if is_train:
                assert 0 <= pid2label[pid] <= 55, f"Invalid train PID: {pid}"
            else:
                assert 0 <= pid2label[pid] <= 4, f"Invalid test PID: {pid}"

            if is_train:
                pid = pid2label[pid]

            data.append(
                Datum(
                    img_path=img_path,
                    pid=pid,
                    camid=camid
                )
            )
            
            print(Datum(img_path=img_path, pid=pid, camid=camid))
            exit()

        return data