
"""Four real gloo ranks. Run standalone; prints JSON on rank 0."""
import json, os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, N = 4, 256


def worker(rank, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29517")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)

    torch.manual_seed(100 + rank)
    x = torch.randn(N)

    ar = x.clone()
    dist.all_reduce(ar, op=dist.ReduceOp.SUM)
    ar /= WORLD

    rs = torch.empty(N // WORLD)
    dist.reduce_scatter_tensor(rs, x.clone(), op=dist.ReduceOp.SUM)
    rs /= WORLD

    ag = torch.empty(N)
    dist.all_gather_into_tensor(ag, rs.contiguous())

    if rank == 0:
        out["all_reduce"] = ar.tolist()
        out["all_gather_of_reduce_scatter"] = ag.tolist()
        out["inputs"] = None
    dist.destroy_process_group()


if __name__ == "__main__":
    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(worker, args=(out,), nprocs=WORLD, join=True)
    inputs = []
    for r in range(WORLD):
        torch.manual_seed(100 + r)
        inputs.append(torch.randn(N).tolist())
    print(json.dumps({"all_reduce": out["all_reduce"],
                      "all_gather_of_reduce_scatter": out["all_gather_of_reduce_scatter"],
                      "inputs": inputs}))
