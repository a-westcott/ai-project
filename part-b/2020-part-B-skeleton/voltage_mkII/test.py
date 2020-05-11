from multiprocessing import Pool
import time
import datetime
import random
import numpy as np

from state import State
from features import Φ, H, INF

a = np.array([
    [0,0,0,0,0,0,0,0],
    [0,-1,0,0,0,0,0,0],
    [0,0,-1,1,0,0,0,0],
    [0,0,0,1,-1,-1,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,-1,0,0,0,0,1,0],
    [0,0,0,0,-3,0,1,0]
])


s = State(a)

θ = np.load('w_opn.npy')


def negamax(state, alpha, beta, depth, θ, memoised_states=None):
    if state.training_terminal_test():
        #print(state)
        #print(state.utility())
        #print(depth)
        return state.utility(train_end=True)
    if depth == 0:
        if memoised_states:
            return H(Φ(state, memoised_states), θ)
        return H(Φ(state), θ)

    v = -INF*10
    for a in state.actions():
        child = state.result(a)
        nmax = -negamax(child, -beta, -alpha, depth-1, θ, memoised_states)
        v = max(v, nmax)
        if depth ==1:
            pass #print(v, nmax, a)
        alpha = max(alpha, v)
        if alpha >= beta:
            return v
                
    return v


def negamax2(state, alpha, beta, depth, θ, memoised_states=None):
    if state.training_terminal_test():
        return state.utility(train_end=True)
    if depth == 0:
        if memoised_states:
            return H(Φ(state, memoised_states), θ)
        return H(Φ(state), θ)

    v = -INF*4
    for a in state.actions():
        child = state.result(a)
        nmax = -negamax2(child, -beta, -alpha, depth-1, θ, memoised_states)
        v = max(v, nmax)
        
        if v >= beta:
            return v
        alpha = max(alpha, v) 

    return v


def main():
    #print(s)
    #print(s.actions())
    s1 = State(s.board*-1)
    #print()
    #print(negamax2(s, -INF*10, INF*10, 2, θ))
    print()
    print(s1)
    print()
    #print(negamax2(s1, -INF*4, INF*4, 2, θ))

    depth =2

    #v = negamax2(s1, -INF*4, INF*4, 2, θ)



    best_action = None
    alpha, beta, v = -4*INF, 4*INF, -4*INF
    for a in s1.actions():
        child = s1.result(a)
        #child.board *= -1
        nmax = -negamax2(child, -beta, -alpha, depth-1, θ)
        if nmax > alpha:
            alpha = nmax
            best_action = a

    
    print(best_action)


def main2():
    s1 = State(s.board*-1)
    print(s1.actions())

if __name__ == '__main__':
    main()