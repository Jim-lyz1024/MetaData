import copy
import random
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size equals N*K.
    """
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        
        # 构建索引字典 {pid: [indices]}
        self.pid_to_indices = {}
        for idx, item in enumerate(data_source):
            pid = item.pid
            if pid not in self.pid_to_indices:
                self.pid_to_indices[pid] = []
            self.pid_to_indices[pid].append(idx)
            
        self.pids = list(self.pid_to_indices.keys())
        
        # 计算每个pid需要的batch数
        self.pid_batches = {}
        for pid in self.pids:
            indices = self.pid_to_indices[pid]
            num_batches = len(indices) // num_instances
            if num_batches == 0:  # 如果实例数不足num_instances
                num_batches = 1
            self.pid_batches[pid] = num_batches
            
        # 计算一个epoch的总长度
        self.length = sum(self.pid_batches.values()) * num_instances

    def __iter__(self):
        final_idxs = []
        
        # 为每个pid创建实例索引
        pid_indices = {}
        for pid in self.pids:
            indices = self.pid_to_indices[pid]
            if len(indices) < self.num_instances:
                indices = np.random.choice(indices, size=self.num_instances, replace=True)
            else:
                indices = np.random.choice(indices, size=len(indices), replace=False)
            pid_indices[pid] = list(indices)
        
        # 打乱pid顺序
        pids = np.random.permutation(self.pids)
        
        # 为每个batch选择pids
        current_pids = []
        for pid in pids:
            current_pids.append(pid)
            if len(current_pids) == self.num_pids_per_batch:
                # 为每个选中的pid采样num_instances个实例
                for pid in current_pids:
                    if len(pid_indices[pid]) < self.num_instances:
                        selected_idxs = np.random.choice(
                            self.pid_to_indices[pid],
                            size=self.num_instances,
                            replace=True
                        )
                    else:
                        selected_idxs = pid_indices[pid][:self.num_instances]
                        pid_indices[pid] = pid_indices[pid][self.num_instances:]
                        
                    final_idxs.extend(selected_idxs)
                current_pids = []
        
        # 处理剩余的pids
        if len(current_pids) > 0:
            for pid in current_pids:
                if len(pid_indices[pid]) < self.num_instances:
                    selected_idxs = np.random.choice(
                        self.pid_to_indices[pid],
                        size=self.num_instances,
                        replace=True
                    )
                else:
                    selected_idxs = pid_indices[pid][:self.num_instances]
                final_idxs.extend(selected_idxs)
        
        # 确保所有索引都是Python int类型
        final_idxs = [int(idx) for idx in final_idxs]
        
        return iter(final_idxs)

    def __len__(self):
        return self.length