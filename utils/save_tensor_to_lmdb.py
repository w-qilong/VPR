import torch
import numpy as np
import lmdb
import pickle

def save_tensor_to_lmdb(tensors, lmdb_path, map_size=1099511627776):
    """
    将shape为[b,1,n,d]的tensor存储到LMDB中
    
    Args:
        tensors: shape为[b,1,n,d]的torch tensor
        lmdb_path: LMDB数据库路径
        map_size: LMDB数据库大小限制（默认1TB）
    """
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        for idx in range(tensors.shape[0]):
            # 获取单个样本并转换为numpy数组
            tensor = tensors[idx].cpu().numpy()
            # 使用pickle序列化数据
            serialized_tensor = pickle.dumps(tensor)
            # 存储数据，键为字符串格式的索引
            txn.put(str(idx).encode(), serialized_tensor)
    
    env.close()

def load_tensor_from_lmdb(lmdb_path, batch_size=None):
    """
    从LMDB中读取tensor数据
    
    Args:
        lmdb_path: LMDB数据库路径
        batch_size: 要读取的批次大小，若为None则读取所有数据
    
    Returns:
        torch tensor，shape为[b,1,n,d]
    """
    env = lmdb.open(lmdb_path, readonly=True)
    
    with env.begin() as txn:
        # 获取数据库中的样本数量
        num_samples = int(txn.stat()['entries'])
        if batch_size is None:
            batch_size = num_samples
            
        tensors = []
        for idx in range(batch_size):
            # 读取序列化数据
            serialized_tensor = txn.get(str(idx).encode())
            if serialized_tensor is None:
                break
            # 反序列化数据
            tensor = pickle.loads(serialized_tensor)
            # 转换为torch tensor
            tensor = torch.from_numpy(tensor)
            tensors.append(tensor)
    
    env.close()
    # 将所有tensor拼接成一个batch
    return torch.stack(tensors, dim=0)

# 存储数据
tensor = torch.randn(10, 1, 100, 64)  # 示例tensor
print(tensor[0])
save_tensor_to_lmdb(tensor, "/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/lmdb/test.lmdb")

# 读取数据
loaded_tensor = load_tensor_from_lmdb("/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/lmdb/test.lmdb")
# 或者指定batch_size读取
loaded_batch = load_tensor_from_lmdb("/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/lmdb/test.lmdb", batch_size=5)

print(loaded_tensor.shape)
print(loaded_batch.shape)
print(loaded_tensor[0])

