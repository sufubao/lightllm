# Adapted from triton.autotune

from typing import Dict
from triton.runtime.jit import KernelInterface
import inspect
import os


class Autotuner(KernelInterface):

    def __init__(
        self,
        fn,
        configs,
        key_func,
        warmup=25,
        rep=100,
    ):
        self.fn = fn
        self.configs = configs
        self.key_func = key_func
        self.warmup = warmup
        self.rep = rep

        self.cache = {}

    def search_best_config(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        key = self.key_func(*args, **kwargs)
        if key not in self.cache:
            best_config = self.search_best_config(key, *args, **kwargs)
        else:
            best_config = self.cache[key]
        kwargs["config"] = best_config
        return self.fn(*args, **kwargs)


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


def autotune(configs, key_func_dict={}, warmup=25, rep=100):
    def decorator(fn):
        return Autotuner(fn, configs, generate_function_from_dict(key_func_dict), warmup=warmup, rep=rep)

    return decorator