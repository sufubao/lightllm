import inspect
from functools import wraps

def print_key_decorator(key_dict=None):
    if key_dict is None:
        key_dict = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数的参数名称
            signature = inspect.signature(func)
            parameters = signature.parameters

            # 用于存储格式化后的结果
            output = []

            # 遍历所有参数，按顺序输出
            idx = 0
            for param_name, param in parameters.items():
                if param.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
                    # 处理位置参数（args）
                    if idx < len(args):  # 如果args中有这个位置参数
                        value = args[idx]
                        idx += 1
                    else:  # 如果没有传入args值，则取默认值
                        value = param.default
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    # 处理关键字参数（kwargs）
                    value = kwargs.get(param_name, param.default)
                else:
                    value = param.default
                
                # 格式化输出
                if param_name in key_dict:
                    output.append(key_dict[param_name](value))
                else:
                    output.append(f"{param_name}={value}")

            # 输出格式化后的内容，按逗号分隔
            print(",".join(output))

            # 调用原始函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# 使用装饰器
@print_key_decorator(
    key_dict={
        'b': lambda x: f"len1={x.shape[0]},len2={x.shape[1]}",
        "c": lambda x: f"lk={x.shape[-1]}",
        "d": lambda x: f"d={x}",
        'e': lambda x: f"e={x}"
    }
)
def foo(a, b, c, d=1, e=2):
    pass

# 示例调用（假设b, c等是NumPy数组或者具有类似形状的对象）
import numpy as np

b = np.array([[1, 2], [3, 4]])  # shape (2, 2)
c = np.array([1, 2, 3])         # shape (3,)
foo(10, b, c, d=3, e=4)
