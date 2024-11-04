import os
import os.path as osp

__all__ = ['Datum', 'ReIDDataset'] # Explicitly export these classes

class Datum:
    """Data instance for ReID which defines basic attributes.
    
    Args:
        img_path (str): Image path
        pid (int): Person ID
        camid (int): Camera ID
        is_train (bool): Training data or test data
    """
    def __init__(self, img_path, pid, camid, is_train=True):
        self._img_path = img_path
        self._pid = pid 
        self._camid = camid
        self._is_train = is_train

    @property
    def img_path(self):
        return self._img_path
    
    @property
    def pid(self):
        return self._pid
        
    @property
    def camid(self):
        return self._camid
        
    @property
    def is_train(self):
        return self._is_train

class ReIDDataset:
    """Base class for ReID datasets."""
    def __init__(self, root='', verbose=True):
        self.root = os.path.normpath(str(root))
        self.dataset_dir = os.path.normpath(os.path.join(self.root, self.dataset_dir))
        self.train_dir = os.path.normpath(os.path.join(self.dataset_dir, 'train'))
        self.query_dir = os.path.normpath(os.path.join(self.dataset_dir, 'query'))
        self.gallery_dir = os.path.normpath(os.path.join(self.dataset_dir, 'gallery'))
        
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Train directory: {self.train_dir}")
        print(f"Query directory: {self.query_dir}")
        print(f"Gallery directory: {self.gallery_dir}")
        
        self._check_before_run()
        
        # Process data
        self.train = self._process_train(self.train_dir)
        self.query = self._process_test(self.query_dir)
        self.gallery = self._process_test(self.gallery_dir) 
        
        if verbose:
            self.print_dataset_statistics()
            
        # Get meta info
        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
            
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
        """Process training data - to be implemented by subclass"""
        raise NotImplementedError
        
    def _process_test(self, dir_path):
        """Process test data - to be implemented by subclass"""
        raise NotImplementedError
        
    def get_num_pids(self, data):
        """Get number of unique person IDs"""
        pids = set()
        for item in data:
            pids.add(item.pid)
        return len(pids)
        
    def get_num_cams(self, data):
        """Get number of unique cameras"""
        cams = set()
        for item in data:
            cams.add(item.camid)
        return len(cams)
        
    def print_dataset_statistics(self):
        """Print dataset statistics"""
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(
            self.get_num_pids(self.train),
            len(self.train), 
            self.get_num_cams(self.train)
        ))
        print("  query    | {:5d} | {:8d} | {:9d}".format(
            self.get_num_pids(self.query),
            len(self.query),
            self.get_num_cams(self.query)
        ))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(
            self.get_num_pids(self.gallery), 
            len(self.gallery),
            self.get_num_cams(self.gallery)
        ))
        print("  ----------------------------------------")