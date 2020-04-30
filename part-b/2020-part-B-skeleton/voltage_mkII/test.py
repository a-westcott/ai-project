from multiprocessing import Pool
import time
import datetime
import random

from state import State
from features import Φ


a = {}

def f(n):
    print(f'doing {n}!')
    for i in range(10000000):
        n**2
    return n+1

def main2():
    start = datetime.datetime.now()
    for i in range(1000000):
        Φ(State())
    print(datetime.datetime.now() - start)

def main():
    nums = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
    with Pool(8) as p:
        print(p.map(f, nums))
    print('what do we end up with?')
    print(a)


if __name__ == '__main__':
    main2()