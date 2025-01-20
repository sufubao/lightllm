# Adapted from triton.autotune
from typing import Dict
from triton.runtime.jit import KernelInterface
from triton.runtime.errors import OutOfResources
from triton.testing import do_bench, do_bench_cudagraph
from triton.compiler.errors import CompileTimeAssertionFailure
import ujson as json
import os 
import torch
import inspect
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import get_current_device_name
import itertools


logger = init_logger(__name__)

def generate_configs(config_dict):
    # 获取所有键和对应的值列表
    keys = config_dict.keys()
    values = config_dict.values()
    
    # 使用 itertools.product 生成所有可能的组合
    combinations = itertools.product(*values)
    
    # 将每个组合转换为字典
    configs = [dict(zip(keys, combination)) for combination in combinations]
    
    return configs


def generate_getkey_function(fn, key_func_dict={}):
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


def get_cache_file_path(kernel_name, static_key):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, "configs", kernel_name)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    device_name = get_current_device_name().replace(" ", "_")
    file_name = f"{static_key},{device_name}.json"
    return os.path.join(cache_dir, file_name)


def load_configs(config_file):
    if not os.path.exists(config_file):
        logger.warning(f"Config file {config_file} does not exist")
        return None
    with open(config_file, "r") as f:
        return json.load(f)


class Autotuner():

    def __init__(self, kernel_name, fn, config_dict, static_key_dict, dynamic_key_dict, enable_autotune, warmup, rep, top_k, init_warmup, init_rep, use_cuda_graph):
        self.kernel_name = kernel_name
        self.fn = fn
        self.configs = generate_configs(config_dict)
        self.static_key = None
        self.static_key_func = generate_getkey_function(fn, static_key_dict)
        self.dynamic_key_func = generate_getkey_function(fn, dynamic_key_dict)
        self.warmup = warmup
        self.rep = rep
        self.top_k = min(top_k, len(self.configs))
        self.init_warmup = init_warmup
        self.init_rep = init_rep
        self.enable_autotune = enable_autotune

        if use_cuda_graph:
            self.cuda_stream = torch.cuda.Stream()
        self.use_cuda_graph = use_cuda_graph

        self.know_rank_id = False
        self.cache_file_path = None
        self.cache = None

    @staticmethod
    def _load_cache(cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file_path, "w") as f:
            json.dump(self.cache, f, indent=4)
        
    def _bench(self, warmup, rep ,config, *args, **kwargs):
        kwargs["config"] = config
        def kernel_call():
            try:
                self.fn(*args,**kwargs)
            except Exception as e:
                raise
        try:
            if self.use_cuda_graph:
                with torch.cuda.stream(self.cuda_stream):
                    bench_res = do_bench_cudagraph(kernel_call, rep=rep, return_mode="median")
                return bench_res
            return do_bench(kernel_call, warmup=warmup, rep=rep, quantiles=(0.5,))[0]
        except (OutOfResources, CompileTimeAssertionFailure):
            return float("inf")

    def search_best_config(self, key, *args, **kwargs):
        # TODO: Optimize the implementation
        initial_results = []
        for config in tqdm(self.configs, desc="Initial screening"):
            bench_time = self._bench(self.init_warmup, self.init_rep, config, *args, **kwargs)
            initial_results.append((config, bench_time))
        
        initial_results.sort(key=lambda x: x[1])
        top_k_configs = [config for config, _ in initial_results[:self.top_k]]
        
        final_results = []
        for config in tqdm(top_k_configs, desc="Fine-grained screening"):
            bench_time = self._bench(self.warmup, self.rep, config, *args, **kwargs)
            final_results.append((config, bench_time))
        
        final_results.sort(key=lambda x: x[1])
        self.cache[key] = final_results[0][0]
        self._save_cache()

    def run(self, *args, **kwargs):
        if not self.know_rank_id:
            # First time to run the function, 
            # get the rank id, only rank 0 will do the autotune
            # get the static key and locate the cache file
            # TODO: Support multi-node autotune
            self.know_rank_id = True
            # assert dist.is_initialized(), "Distributed environment is not initialized"
            if dist.is_initialized():
                rank_id = dist.get_rank()
            else:
                rank_id = 0
            if rank_id != 0:
                self.enable_autotune = False
            self.static_key = self.static_key_func(*args, **kwargs)
            self.cache_file_path = get_cache_file_path(self.kernel_name, self.static_key)
            self.cache = self._load_cache(self.cache_file_path)

        key = self.dynamic_key_func(*args, **kwargs)
        if key not in self.cache:
            if self.enable_autotune:
                self.search_best_config(key, *args, **kwargs)
            else:
                self.cache[key] = None
        kwargs["config"] = self.cache[key]
        return self.fn(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def autotune(kernel_name, config_dict, static_key_dict={}, dynamic_key_dict={}, warmup=25, rep=100, init_warmup=2, init_rep=3, top_k=10, use_cuda_graph=True):
    # 0: 关闭 autotune，1: 使用 autotune 配置，2: 启用 autotune 
    autotune_level = os.getenv("AUTOTUNE_LEVEL", "1")
    assert autotune_level in ["0", "1", "2"], "Invalid value for environment variable AUTOTUNE_LEVEL"

    def decorator(fn):
        if autotune_level == "0":
            def wrapper(*args, **kwargs):
                kwargs["config"] = None  # 保持配置为 None
                return fn(*args, **kwargs)
            return wrapper
        
        enable_autotune = autotune_level == "2"

        # 正常返回 Autotuner 实例
        return Autotuner(kernel_name, fn, config_dict, static_key_dict, dynamic_key_dict, enable_autotune, warmup=warmup, rep=rep, init_warmup=init_warmup, init_rep=init_rep, 
                         top_k=top_k, use_cuda_graph=use_cuda_graph)
    return decorator