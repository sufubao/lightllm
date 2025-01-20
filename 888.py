import itertools

def generate_configs(config_dict):
    # 获取所有键和对应的值列表
    keys = config_dict.keys()
    values = config_dict.values()
    
    # 使用 itertools.product 生成所有可能的组合
    combinations = itertools.product(*values)
    
    # 将每个组合转换为字典
    configs = [dict(zip(keys, combination)) for combination in combinations]
    
    return configs

# 示例使用
config_dict = {
    "BLOCK_M": [64, 128],
    "BLOCK_N": [64, 128],
    "num_warps": [4, 8],
    "num_stages": [1, 2],
}

configs = generate_configs(config_dict)
for config in configs:
    print(config)