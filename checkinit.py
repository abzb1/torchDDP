import torch as torch
from torch import distributed as dist
import os
import time
from datetime import timedelta

# decorator for time analysis
def printtime(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total = round((end - start) * 1000, 4)
        print(f"{os.environ['SLURM_PROCID']}, {func.__name__}, {start}, {end}, {total}")
        return result
    return wrapper

@printtime
# availability check func
def areUReady():
    print("Are you ready?")
    assert dist.is_available(), "I'm not Ready"
    print("Yes, I am.")

@printtime
# make random tensor func
def makeTensor(size, w, c, *args, **kwargs) ->"torch.tensor.cuda()" :
    tensor = torch.rand(size, c, w, w, *args, **kwargs)
    tensor = tensor.cuda()
    return tensor

# initiate Process group func
@printtime
def setup(backend:"str backend", rank, size):
    print(f"Initiating backend : {backend}")
    os.environ['MASTER_ADDR'] = "dolphin"
    os.environ['MASTER_PORT'] = "29500"
    os.environ['NCCL_NET_GDR_LEVEL'] = "4"

    # debug option
    #os.environ['NCCL_DEBUG'] = "INFO"
    #os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
    
    # init

    os.environ['NCCL_BLOCKING_WAIT'] = "1"
    block_time = timedelta(days = 0, hours = 0, minutes = 10)
    dist.init_process_group(dist.Backend(backend), timeout = block_time, rank = rank, world_size = size)

@printtime
def cleanup():
    dist.destroy_process_group()
