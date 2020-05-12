try:
    from state import State, MOVE, ALL, INF
except:
    from voltage_alt.state import State, MOVE, ALL, INF

import numpy as np
import gc
from math import tanh

ALL_STACKS = 0

R3 = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
      (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7),
      (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
      (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
      
R2 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
      (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
      (2, 1), (3, 1), (4, 1), (5, 1), 
      (2, 6), (3, 6), (4, 6), (5, 6)]   

R1 = [(2, 2), (2, 3), (2, 4), (2, 5),
      (5, 2), (5, 3), (5, 4), (5, 5),
      (3, 2), (4, 2), (3, 5), (4, 5)]
  
R0 =[(3, 3), (3, 4),  
     (4, 3), (4, 4)]

RINGS = [R0, R1, R2, R3]

BUFFER = [(0, 2), (2, 2), (1, 0), (-1, -2), (-2, -1), (-2, -2), (-1, -1), (-2, 1), (-1, 1), 
          (0, -1), (0, -2), (0, 1), (2, -2), (2, -1), (1, 2), (2, 1), (-2, 0), (-1, 0), 
          (-2, 2), (0, 0), (1, 1), (2, 0), (1, -2), (1, -1), (-1, 2)]



def H(features, θ):
    h = np.dot(features, θ)
    if h > 0.99*INF:
        return INF*tanh(h/35)
    if h < -0.99*INF:
        return -INF*tanh(h/35)
    return h


def Φ(state, memoized_states=None): 
    if memoized_states is not None and state in memoized_states:
        return memoized_states[state]

    X, O = 1, 0
    board = state.board
    opp_b = State(-1*board).board
    X_stacks = [(x, y) for x, y in ALL if board[x][y] > 0]
    O_stacks = [(x, y) for x, y in ALL if board[x][y] < 0]
    #X_stacks_by_size = [[(x, y) for x, y in X_stacks if board[x][y] ==  stack_size] for stack_size in range(1, 13)]
    #O_stacks_by_size = [[(x, y) for x, y in O_stacks if board[x][y] == -stack_size] for stack_size in range(1, 13)]

    def largest_connected_cluster(player):
        '''
        largest connected cluster in terms of number of stacks
        '''
        NORM = 12
        player_stacks = X_stacks.copy() if player == X else O_stacks.copy()
        colour = 1 if player == X else -1
        adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        
        largest_connected_cluster = 0
        num_stacks = len(player_stacks) 

        while len(player_stacks) > 0:
            cur_piece = player_stacks[0]
            x,y = cur_piece
            are_adj = set()
            checked_adj = set()
            are_adj.add((x,y))
            while len(are_adj) > len(checked_adj):
                x,y = are_adj.difference(checked_adj).pop()
                for d in adj:
                    dx, dy = x+d[0], y+d[1]
                    if 0 <= dx < 8 and 0 <= dy < 8:
                        if board[dx][dy]*colour > 0:
                            are_adj.add((dx, dy))
                checked_adj.add((x,y))
                player_stacks.remove((x,y))
            if len(are_adj) > largest_connected_cluster:
                largest_connected_cluster = len(are_adj)
                if largest_connected_cluster >= num_stacks/2:
                    return largest_connected_cluster/NORM

        return largest_connected_cluster/NORM

    def mobility(player):
        ''' How many different squares the player can move onto '''
        NORM = 1
        player_stacks = X_stacks if player == X else O_stacks
        colour = 1 if player == X else -1

        contribution = 0
        for x, y in player_stacks:
            stack_size = abs(board[x][y])
            num_spots = 0
            for dx in range(1, stack_size+1):
                if 0 <= x+dx < 8 and ((board[x+dx][y] == 0) or (board[x+dx][y]*colour > 0)):
                    num_spots += 1
                if 0 <= x-dx < 8 and ((board[x-dx][y] == 0) or (board[x-dx][y]*colour > 0)):
                    num_spots += 1

            for dy in range(1, stack_size+1):
                if 0 <= y+dy < 8 and ((board[x][y+dy] == 0) or (board[x][y+dy]*colour > 0)):
                    num_spots += 1
                if 0 <= y-dy < 8 and ((board[x][y-dy] == 0) or (board[x][y-dy]*colour > 0)):
                    num_spots += 1
                
            # contribution += num spots that piece can move / num_spaces it could move if free
            contribution += num_spots / (4*stack_size)
        return contribution / len(player_stacks)/NORM

    def pieces(player, board=board):
        '''
        Returns the number of pieces on a board for the current player.
        Defaults to the board of the current state, can pass in a different board.
        '''
        NORM = 12
        if player == X:
            return board[board > 0].sum()/NORM
        return -board[board < 0].sum()/NORM
    
    def stacks(player, board=board):
        ''' 
        Takes a player and returns the number of stacks that player has INT 
        Defaults to the board of the current state, can pass in a different board.
        '''
        NORM = 12
        if player == X:
            return (board > 0).sum()/NORM
        return (board < 0).sum()/NORM
    
    def actions(player):
        ''' Returns the number of actions the player has INT'''
        NORM = 130
        if player == X:
            return len(State(board).actions())/NORM
        return len(State(opp_b).actions())/NORM
    
    def connectivity(player):
        NORM = 8
        player_stacks = X_stacks if player == X else O_stacks
        colour = 1 if player == X else -1
        adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        
        count = 0
        s = set()
        for x, y in player_stacks:
            for d in adj:
                dx, dy = x+d[0], y+d[1]
                if 0 <= dx < 8 and 0 <= dy < 8:
                    if board[dx][dy]*colour > 0:
                        s.add((dx, dy))
        return len(s)/NORM
            
    def threat(player):
        NORM = 8
        player_stacks = X_stacks if player == X else O_stacks
        colour = 1 if player == X else -1
        adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        
        count = 0
        s = set()
        for x, y in player_stacks:
            for d in adj:
                dx, dy = x+d[0], y+d[1]
                if 0 <= dx < 8 and 0 <= dy < 8:
                    if board[dx][dy]*colour < 0:
                        s.add((dx, dy))
        return len(s)/NORM
            
    # Distance to opponent
    def closest_opposing_pieces(player):
        NORM = 12
        closest = -16
        for Xx, Xy in X_stacks:
            for Ox, Oy in O_stacks:
                dist = abs(Xx - Ox) + abs(Xy - Oy)
                if dist < closest:
                    if dist == 1:
                        return closest
                    closest = dist
        return closest/NORM

    def avg_closeness(player):
        NORM = 12
        running_total = 0
        our_stacks, opp_stacks = (X_stacks, O_stacks) if player == X else (O_stacks, X_stacks)
         
        for x1, y1 in our_stacks:
            closest = 16
            for x2, y2 in opp_stacks:
                dist = abs(x1 - x2) + abs(y1 - y2)
                if dist < closest:
                    if dist == 1:
                        closest = dist
                        break
                    closest = dist
            running_total += closest

        return running_total/len(X_stacks)/NORM

    def ratio():
        return pieces(X) / pieces(O)

    fs = [largest_connected_cluster,
           mobility, pieces, stacks, actions, connectivity, threat, closest_opposing_pieces, avg_closeness]

    features = [f(player) for f in fs for player in [X, O]]
    diffs = []
    for i in range(0, len(features), 2):
        diffs.append(features[i] - features[i+1])
    features = [ratio()] + features+diffs
    
    piece_adv = (np.sign(pieces(X) - pieces(O)) + 1)//2

    num_features = len(features)
    for i in range(num_features):
        features.append(piece_adv * features[i])
        features.append((1-piece_adv) * features[i])
    
    features = np.array(features)
    if memoized_states is not None:
        memoized_states[state] = features
    return features

num_features = len(Φ(State(), {}))

def main():
    state = State()

if __name__ == '__main__':
    main()