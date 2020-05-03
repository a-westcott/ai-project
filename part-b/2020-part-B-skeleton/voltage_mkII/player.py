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

try:
    from state import State 
    from features import Φ, H, RINGS, R0, R1, R2, R3, INF
    from opening import opening_book
    from endgame import n_v_one, NvTwo

except:
    from voltage_mkII.state import State
    from voltage_mkII.features import Φ, H, RINGS, R0, R1, R2, R3, INF
    from voltage_mkII.opening import opening_book
    from voltage_mkII.endgame import n_v_one, NvTwo

class BasePlayer:
    def __init__(self, colour):
        self.colour = colour
        self.state = State()
        self.n_v_two = None
        self.θo = np.array([-0.03245203,  0.99824037,  0.22783496, -0.21423782,  0.87932015,
       -1.51462336,  0.03906433, -0.23706977,  0.34055689,  0.07981415,
       -0.20269713,  0.13594931,  0.4811231 , -0.25081762,  0.08269731,
        0.17427769,  0.17046138,  0.00552456,  0.18165575,  0.03117019,
        0.22360666,  0.29112818,  0.09886535,  0.07394381,  0.14480535,
       -0.0458725 ,  0.17789136,  0.02259329,  0.18554085,  0.25032028,
        0.19979095,  0.04382655, -1.1595476 ,  0.07027167,  1.28326505,
        0.49815301, -0.06775797, -0.56555695,  0.77952216, -0.09158038,
        0.16493682,  0.21380222, -0.13147541, -0.05757411,  0.19067786,
        0.15529807, -0.07093985,  0.2343219 ])
        self.θd = np.array([-0.08768132,  0.23236633,  0.41423004,  0.0533873 ,  1.38372173,
       -1.50295447,  0.26323445, -0.45562112,  0.23581217, -0.14934506,
       -0.12273668,  0.28583526,  0.33958737, -0.0664413 ,  0.0287397 ,
        0.07734851,  0.17538832,  0.00511528,  0.07038506,  0.14925856,
        0.02994492, -0.2161816 ,  0.03040666, -0.31272627,  0.15439061,
       -0.08338026,  0.05915729,  0.09207598,  0.0320818 , -0.21658966,
        0.03491684, -0.24772718, -0.20893653, -0.00515285,  1.79232475,
        0.73616766,  0.27404612, -0.29746083,  0.29491756, -0.04860881,
        0.17027304, -0.07887349,  0.24612653,  0.3432183 ,  0.23777087,
       -0.03291868,  0.24867146,  0.28264402])
        self.θm = np.array([-7.49866922e-02,  2.10564551e-01,  5.53065616e-01,  8.71081905e-02,
        1.51249217e+00, -1.59468861e+00,  3.52465386e-01, -4.78696991e-01,
        2.87707285e-01, -3.19673815e-01, -1.24112746e-01,  2.31974319e-01,
        3.45404688e-01,  2.35514550e-02,  5.09605859e-02,  6.73170539e-02,
        1.38081264e-01, -9.44593392e-02,  5.69468259e-02,  1.01065756e-01,
        1.84625511e-01, -4.59254339e-01,  2.17274583e-02, -3.09295743e-02,
        1.33464934e-01, -1.23598387e-01,  5.42143680e-02,  6.33140197e-02,
        1.65477888e-01, -4.07786462e-01, -6.91803856e-04, -1.06261624e-02,
       -1.74440132e-01,  1.32624093e-01,  1.99606967e+00,  8.31162377e-01,
        4.96269990e-01, -2.44975954e-01,  2.10742122e-01, -1.63564680e-02,
        2.32540604e-01, -4.41189302e-02,  6.43879850e-01,  5.26570326e-02,
        2.57063321e-01, -9.09965173e-03,  5.73264349e-01,  9.93435858e-03])
        self.θe = np.array([  1.56560706,   3.96900362,   5.11346721,  -1.12599543,
        11.48095028, -16.99407079,  10.2035231 , -15.72219639,
         5.08877306,  -7.38571562,  -1.01366041,   4.36070018,
         8.57794824,   2.9960656 ,   2.13875379,   2.16813698,
         0.31801509,  -0.16669887,   2.72781055,  -1.9823687 ,
         5.61201157,  -8.15149158,   3.05580748,  -4.82476029,
         0.31305351,  -0.18471794,   2.78409382,  -2.01513377,
         5.37560988,  -8.05207927,   3.07085662,  -4.79597403,
        -2.63588364,   8.33490444,  28.13587765,  26.92425107,
        12.25256177,  -5.45846117,   6.4831445 ,  -0.16450443,
         0.48465725,   4.62552765,  13.93078778,   7.94802021,
         0.52214088,   4.71457699,  13.7233278 ,   7.9130933 ])
    def action(self):
        if self.state.turn < 8:
            try:
                pass
                return tuple(opening_book[str(self.state.board)])
            except:
                pass
                #print('Failed to get opening move')
                #assert(False)
        
        # n v one endgame
        if self.state.board[self.state.board < 0].sum() == -1 and self.state.board[self.state.board > 0].sum() > 1:
            return self.format_action(n_v_one(self.state))

        # n v two endgame
        if self.state.board[self.state.board < 0].sum() == -2 and self.state.board[self.state.board > 0].sum() > 2:
            if self.n_v_two is None:
                self.n_v_two = NvTwo()
            return self.format_action(self.n_v_two.move(self.state))


        actions = []
        depth = 2
        if self.state.stage[0] == OPN:
            θ = self.θo
        elif self.state.stage[0] == DEV:
            θ = self.θd
        elif self.state.stage[0] == MID:
            θ = self.θm
        else:
            θ = self.θe
            depth = 2*depth


        alpha, beta, v = -INF, INF, -INF
        for a in self.state.actions():
            child = self.state.result(a)
            nmax = (-self.negamax(State(-1*child.board), -beta, -alpha, depth-1, θ))
            actions.append((nmax, a))
            v = max(v, nmax)
            alpha = max(alpha, v)
        
        return self.format_action(max(actions)[1])

    def update(self, colour, action):    
        self.state = self.state.result(action) 

        
        # invert the sign of the pieces so that positive has the next move
        self.state.board = -1*self.state.board

    def format_action(self, action):
        if action[0] == BOOM:
            move, orig = action
            statement = (BOOM, orig)
        else:
            move, n, orig, dest = action
            statement = (MOVE, n, orig, dest)
        return statement


    def negamax(self, state, alpha, beta, depth, θ):
        if state.stages_terminal_test():
            return state.utility()
        if depth == 0:
            return H(Φ(state), θ)

        v = -INF
        for a in state.actions():
            child = state.result(a)
            # game state must be flipped
            v = max(v, -self.negamax(State(-1*child.board), -beta, -alpha, depth-1, θ))
            alpha = max(alpha, v)
            if alpha >= beta:
                return v
            
        return v



def H(features, θ):
    h = np.dot(features, θ)
    if h > 0.99*INF:
        return 0.99*INF
    if h < -0.99*INF:
        return -0.99*INF
    return h
