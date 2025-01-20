import inspect

def generate_function_from_dict(fn, key_func_dict={}):
    """
    Dynamically creates a wrapper for the input function, applying specific transformations
    to designated arguments based on key_func_dict.

    Args:
        fn (function): The original function to wrap.
        key_func_dict (dict): A dictionary where keys are argument names from `fn`
                              and values are transformation functions to apply to them.

    Returns:
        function: A new function that applies transformations and returns a comma-separated string
                  of the transformed argument values.
    """
    # Get the argument specification of the function
    spec = inspect.getfullargspec(fn)
    args = spec.args
    defaults = spec.defaults or ()
    
    # Map argument names to their positions
    arg_to_pos = {arg: pos for pos, arg in enumerate(args)}
    
    # Prepare a list of positions and corresponding transformation functions
    processed_params = [(arg_to_pos[key], func) for key, func in key_func_dict.items() if key in arg_to_pos]
    
    # Create a dictionary for default values of the function's arguments
    default_dict = dict(zip(args[-len(defaults):], defaults)) if defaults else {}
    
    def generated_function(*args_call, **kwargs_call):
        """
        The generated wrapper function that applies the transformations.
        """
        # Build a complete list of parameter values
        params = list(args_call)
        for i in range(len(params), len(args)):
            param_name = args[i]
            if param_name in kwargs_call:
                params.append(kwargs_call[param_name])
            elif param_name in default_dict:
                params.append(default_dict[param_name])
            else:
                raise TypeError(f"Missing required argument: {param_name}")
        
        # Apply transformations to specified arguments and collect the results
        result_parts = [func(params[pos]) for pos, func in processed_params]
        
        # Return a comma-separated string of transformed results
        return ",".join(result_parts)
    
    return generated_function

# 示例用法
def foo(a, b, c, d=1, e=2):
    pass

# 生成新函数
new_fn = generate_function_from_dict(foo, {
    'b': lambda x: f"len1={x.shape[0]},len2={x.shape[1]}",
    "c": lambda x: f"lk={x.shape[-1]}",
    "d": lambda x: f"d={x}",
    'e': lambda x: f"e={x}"
})

# 测试
import numpy as np

b_value = np.random.rand(3, 4)
c_value = np.random.rand(5, 6)
d_value = 42

# 调用新函数
result = new_fn(1, b_value, c_value, e=d_value)
assert result == "len=3,lk=6,d=1,e=42"