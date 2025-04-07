import numpy as np

def calculate_memory_usage(descriptors):
    """
    计算一组描述符的内存占用
    :param descriptors: 包含全局和局部描述符的字典
    :return: 内存占用统计结果（单位：MB）
    """
    result = {}
    
    # 统计全局描述符
    if 'global' in descriptors:
        global_desc = descriptors['global']
        result['global'] = {
            'shape': global_desc.shape,
            'dtype': global_desc.dtype,
            'memory': global_desc.nbytes / (1024 * 1024)  # 转换为MB
        }
    
    # 统计局部描述符
    if 'local' in descriptors:
        local_desc = descriptors['local']
        result['local'] = {
            'count': len(local_desc),
            'shape': local_desc[0].shape if len(local_desc) > 0 else None,
            'dtype': local_desc[0].dtype if len(local_desc) > 0 else None,
            'memory': sum(arr.nbytes for arr in local_desc) / (1024 * 1024)  # 转换为MB
        }
    
    # 计算总内存
    total_memory = 0
    if 'global' in result:
        total_memory += result['global']['memory']
    if 'local' in result:
        total_memory += result['local']['memory']
    result['total_memory'] = total_memory
    
    return result

# 示例用法
if __name__ == "__main__":
    # 创建示例描述符
    global_desc = np.random.rand(1024).astype(np.float32)  # 128维全局描述符
    local_descs = [np.random.rand(1024).astype(np.float32) for _ in range(390)]  # 100个128维局部描述符
    
    descriptors = {
        'global': global_desc,
        'local': local_descs
    }
    
    # 计算内存占用
    memory_usage = calculate_memory_usage(descriptors)
    
    # 打印结果
    print("全局描述符:")
    print(f"  形状: {memory_usage['global']['shape']}")
    print(f"  数据类型: {memory_usage['global']['dtype']}")
    print(f"  内存占用: {memory_usage['global']['memory']:.2f} MB")  # 添加单位
    
    print("\n局部描述符:")
    print(f"  数量: {memory_usage['local']['count']}")
    print(f"  单个形状: {memory_usage['local']['shape']}")
    print(f"  数据类型: {memory_usage['local']['dtype']}")
    print(f"  总内存占用: {memory_usage['local']['memory']:.2f} MB")  # 添加单位
    
    print(f"\n总内存占用: {memory_usage['total_memory']:.2f} MB")  # 添加单位