import sys
import time
import numpy as np
import cupy as cp
import torch
from scipy.signal import welch


def allocate_arr(namespace):
    size = 90_000_000
    if namespace == "numpy":
        x = np.zeros(size)
    elif namespace == "cupy":
        x = cp.zeros(size)
    elif namespace == "torch_gpu":
        torch.set_default_device("cuda")
        x = torch.zeros(size)
    elif namespace == "torch_cpu":
        torch.set_default_device("cpu")
        x = torch.zeros(size)
    else:
        raise ValueError(f"unrecognized namespace requested for array backend: {namespace}")
    x[0] = 1
    x[8] = 1
    return x


def main(x):
    f, p = welch(x, nperseg=8)
    print("f:", f)
    print("p:", p)


if __name__ == "__main__":
    # NOTE: had problems interleaving repeats
    # and doing multiple trials with CuPy and torch
    # using different GPU mem alloc strats, etc...
    # so just call this script directly on cmd line
    namespace = sys.argv[1]
    x = allocate_arr(namespace=namespace)
    start = time.perf_counter()
    main(x=x)
    end = time.perf_counter()
    elapsed_time_sec = end - start
    print(f"Elapsed Time for Namespace {namespace} (s):", elapsed_time_sec)
