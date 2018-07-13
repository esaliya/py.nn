"""run.py:"""
# !/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


def run(rank, size):
    """ Distributed function to be implemented later. """
    print(f"I am {rank} of {size}")
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    init_processes(0, 0, run, backend='mpi')
    # size = 2
    # processes = []
    # for rank in range(size):
    #     p = Process(target=init_processes, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()