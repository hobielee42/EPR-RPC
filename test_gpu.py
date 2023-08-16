from numba import jit, cuda
import numpy as np
from datasets import load_dataset

# to measure exec time
from timeit import default_timer as timer

snli = load_dataset("snli")
dataset=snli['test']

# normal function to run on cpu
def func():
    for ex in dataset:
        print(ex['label'])


# function optimized to run on gpu
@jit(target_backend="cuda")
def func2():
    for ex in dataset:
        print(ex['label'])


if __name__ == "__main__":
    start = timer()
    func()
    print("without GPU:", timer() - start)

    start = timer()
    func2()
    print("with GPU:", timer() - start)
