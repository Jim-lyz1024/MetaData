import glob
import re
import os.path as osp
from ..base_dataset import ReIDDataset, Datum  # Add Datum import here
from ..build_dataset import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class ATRW(ReIDDataset):
    """ATRW Dataset.
    
    Reference:
        Li et al. ATRW: A Benchmark for Amur Tiger Re-identification 
        in the Wild. ACM MM 2020.

    Dataset statistics:
        - identities: 149 (train) + 33 (gallery) + 33 (query) 
        - images: 3,730 (train) + 521 (gallery) + 424 (query)
    """
    dataset_dir = 'ATRW'

    def __init__(self, cfg, verbose=True):
        root = cfg.DATASETS.ROOT_DIR
        super(ATRW, self).__init__(root, verbose)

    def _process_train(self, dir_path):
        """Process training data"""
        data = []
        pid_container = set()
        for img_path in glob.glob(osp.join(dir_path, '*.jpg')):
            img_name = osp.basename(img_path)
            # Example filename format: id106_cam01_00159.jpg
            pid = int(img_name.split('_')[0][2:])  # remove 'id' prefix
            camid = int(img_name.split('_')[1][3:])  # remove 'cam' prefix
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in glob.glob(osp.join(dir_path, '*.jpg')):
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0][2:])
            camid = int(img_name.split('_')[1][3:])
            pid = pid2label[pid]
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))

        return data

    def _process_test(self, dir_path):
        """Process test data"""
        data = []
        for img_path in glob.glob(osp.join(dir_path, '*.jpg')):
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0][2:])
            camid = int(img_name.split('_')[1][3:])
            data.append(Datum(img_path=img_path, pid=pid, camid=camid))
            
        return data