BOOM = 'BOOM'
MOVE = 'MOVE'
MAX_TURNS = 250
ALL = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
       (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),
       (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),
       (6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]


OPN, DEV, MID, END = 0, 1, 2, 3

DEPTH = 2

import numpy as np
from copy import deepcopy, copy
from random import shuffle
from collections import defaultdict as dd
from math import tanh

try:
    from state import State 
    from features import Φ, H, RINGS, R0, R1, R2, R3, INF
    from opening import opening_book   
    from endgame import n_v_one, NvTwo
    path = ''

except:
    from voltage_mkII.state import State
    from voltage_mkII.features import Φ, H, RINGS, R0, R1, R2, R3, INF
    from voltage_mkII.opening import opening_book
    from voltage_mkII.endgame import n_v_one, NvTwo
    path = 'voltage_mkII/'

class BasePlayer:
    def __init__(self, colour):
        self.colour = colour
        self.state = State()
        self.n_v_two = None
        self.θo = np.load(path+'w_opn-ab3.npy')

        self.θd = np.load(path+'w_dev-ab3.npy')

        self.θm = np.load(path+'w_mid-ab3.npy')

        self.θe = np.load(path+'w_end-ab3.npy')

    def action(self):
        if self.state.turn < 8 : # and not self.state.turn % 2:
            try:
                if str(self.state.board) in opening_book:
                    return tuple(opening_book[str(self.state.board)])

            except:
                print('Failed to get opening move')
                assert(False)
        
        # n v two endgame
        if self.state.board[self.state.board < 0].sum() == -2 and self.state.board[self.state.board > 0].sum() > 2:
            if self.n_v_two is None:
                self.n_v_two = NvTwo()
            return self.format_action(self.n_v_two.move(self.state))

        # n v one endgame
        if len(self.state.board[self.state.board < 0]) == 1 and self.state.board[self.state.board > 0].sum() > 1:
            return self.format_action(n_v_one(self.state))


        depth = DEPTH
        if self.state.stage[0] == OPN:
            θ = self.θo
        elif self.state.stage[0] == DEV:
            θ = self.θd
        elif self.state.stage[0] == MID:
            θ = self.θm
        else:
            θ = self.θe
            depth = 2*depth


        best_action = None
        alpha, beta = -4*INF, 4*INF
        for a in self.state.actions():
            child = self.state.result(a)
            nmax = -self.negamax(child, -beta, -alpha, depth-1, θ)
            if nmax > alpha:
                alpha = nmax
                best_action = a
        
        return self.format_action(best_action)

    def update(self, colour, action):    
        self.state = self.state.result(action) 
        # invert the sign of the pieces so that positive has the next move
        #self.state.board = -1*self.state.board

    def format_action(self, action):
        if action[0] == BOOM:
            move, orig = action
            statement = (BOOM, orig)
        else:
            move, n, orig, dest = action
            statement = (MOVE, n, orig, dest)
        return statement


    def negamax(self, state, alpha, beta, depth, θ):
        if state.terminal_test():
            return state.utility()
        if depth == 0:
            return H(Φ(state), θ)

        v = -INF
        for a in state.actions():
            child = state.result(a)
            v = max(v, -1*(self.negamax(child, -beta, -alpha, depth-1, θ)))
            if v >= beta:
                return v
            alpha = max(alpha, v)
            
        return v


def H(features, θ):
    h = np.dot(features, θ)
    if h > 0.99*INF:
        return INF*tanh(h/35)
    if h < -0.99*INF:
        return -INF*tanh(h/35)
    return h