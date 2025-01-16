import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 3,               # 类别数
    'eta': 0.1,                   # 学习率
    'max_depth': 4,               # 树的最大深度
    'eval_metric': 'mlogloss'     # 评价指标
}

# 训练模型
bst = xgb.train(params, dtrain, num_boost_round=100)

# 预测
preds = bst.predict(dtest)

# 计算准确率
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy}")


def find_best_config(config_space, bench_function, warmup=10, rep=100):
    import math

    # 初始化当前最佳配置
    current_config = {k: v[0] for k, v in config_space.items()}
    best_config = current_config.copy()
    best_time = float('inf')

    # 协调下降优化
    params = list(config_space.keys())
    while True:
        improved = False
        for param in params:
            # 固定其他参数，优化当前参数
            fixed_params = {k: best_config[k] for k in params if k != param}
            # 二分查找当前参数的最优取值
            values = config_space[param]
            left, right = 0, len(values) - 1
            while left <= right:
                mid = (left + right) // 2
                temp_config = fixed_params.copy()
                temp_config[param] = values[mid]
                # 使用较少的rep进行粗略评估
                time = bench_function(temp_config, warmup, rep // 10)
                # 比较与邻居的运行时间
                if mid - 1 >= left:
                    left_time = bench_function(temp_config, warmup, rep // 10, config={param: values[mid - 1]})
                    if left_time < time:
                        right = mid - 1
                        continue
                if mid + 1 <= right:
                    right_time = bench_function(temp_config, warmup, rep // 10, config={param: values[mid + 1]})
                    if right_time < time:
                        left = mid + 1
                        continue
                # 找到局部最优
                if time < best_time:
                    best_config[param] = values[mid]
                    best_time = time
                    improved = True
                break  # 找到局部最优，退出二分查找
        if not improved:
            break  # 没有进一步优化，退出循环

    # 精细评估最佳配置
    best_time = bench_function(best_config, warmup, rep)

    return best_config
