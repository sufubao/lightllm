import torch
import time
import sys
from typing import List, Dict
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask

logger = init_logger(__name__)


def _init_env(
    args,
    device_index: int,
    nccl_ip,
    nccl_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    import os

    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_MAX_NCHANNELS"] = "2"
    os.environ["NCCL_NSOCKS_PER_CHANNEL"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"
    torch.backends.cudnn.enabled = False

    try:
        # 注册graceful 退出的处理
        from lightllm.utils.graceful_utils import graceful_registry
        import inspect

        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        assert len(mem_managers) == args.tp
        task_out_queue.put("get_mem_managers_ok")
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group(
            "nccl", init_method=f"tcp://{nccl_ip}:{nccl_port}", rank=1, world_size=2, timeout=timedelta(seconds=60)
        )
        task_out_queue.put("nccl_ok")
        while True:
            move_task: KVMoveTask = task_in_queue.get()
            try:
                start = time.time()
                if move_task.move_kv_len != 0:
                    cur_mem = mem_managers[device_index]
                    logger.info(f"trans start: {move_task.to_decode_log_info()}")
                    cur_mem.receive_from_prefill_node(move_task.decode_token_indexes, mem_managers)
                    logger.info(f"trans finished: {move_task.to_decode_log_info()}")
                torch.cuda.synchronize()
                logger.info(f"trans cost time: {(time.time() - start)}, {move_task.to_decode_log_info()}")
                task_out_queue.put("ok")
            except BaseException as e:
                logger.exception(str(e))
                task_out_queue.put("fail")
                raise e
    except BaseException as e:
        logger.exception(str(e))
        sys.exit(-1)
    return


def start_decode_trans_process(
    args,
    device_index: int,
    nccl_ip,
    nccl_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    proc = mp.Process(
        target=_init_env, args=(args, device_index, nccl_ip, nccl_port, task_in_queue, task_out_queue, mem_queues)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"decode trans kv process start, nccl_ip: {nccl_ip}, nccl_port: {nccl_port}")
    return proc