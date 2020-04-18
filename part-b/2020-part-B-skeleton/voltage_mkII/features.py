from player import State, MOVE, ALL
import numpy as np

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


def Φ(state, memoized_states={}): 
    if state in memoized_states:
        return memoized_states[state]

    X, O = 1, 0
    board = state.board
    opp_b = State(-1*board).board
    X_stacks = [(x, y) for x, y in ALL if board[x][y] > 0]
    O_stacks = [(x, y) for x, y in ALL if board[x][y] < 0]
    X_stacks_by_size = [[(x, y) for x, y in X_stacks if board[x][y] ==  stack_size] for stack_size in range(1, 13)]
    O_stacks_by_size = [[(x, y) for x, y in O_stacks if board[x][y] == -stack_size] for stack_size in range(1, 13)]

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

    def largest_almost_connected_cluster_stacks(player, num_pieces=False):
        ''' 
        number of stacks (opt pieces) in an extended cluster
        (vulnerable to one opposing stack in the right spot)
        '''
        NORM = 12
        player_stacks = X_stacks.copy() if player == X else O_stacks.copy()
        max_lost = 0
        adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

        check_spots = set()
        [check_spots.add((x+dx,y+dy)) for x,y in player_stacks for dx,dy in adj if 0 <= x+dx <= 7 and 0 <= y+dy <= 7 and [x+dx,y+dy] not in player_stacks]
        if num_pieces:
            starting_pieces = pieces(player)
        else:
            starting_stacks = stacks(player)
        
        for x,y in check_spots:
            result = state.result(('BOOM', (x,y)))
            if num_pieces:
                lost = starting_pieces - pieces(player, result.board)
            else:
                lost = starting_stacks - stacks(player, board=result.board)
            if lost > max_lost:
                max_lost = lost
        return max_lost/NORM

    def largest_almost_connected_cluster_pieces(player, num_pieces=False):
        ''' 
        number of pieces in an extended cluster
        (vulnerable to one opposing stack in the right spot)
        '''
        # other function is already normalised correctly
        NORM = 1
        return largest_almost_connected_cluster_stacks(player, True)/NORM


    def piece_centrality(player, ring):
        ''' Overall centrality in terms of where pieces are '''
        NORM = 12
        # player_stacks is a list of (x, y) of each stack
        player_stacks = X_stacks if player == X else O_stacks
        colour = 1 if player == X else -1
        
        count = 0
        for pos in player_stacks:
            if pos in ring:
                count += board[pos[0]][pos[1]]
        return count*colour/NORM

    def stack_centrality(player, ring):
        ''' Overall centrality in terms of where stacks are (ignores that bigger stacks have more pieces) '''
        NORM = 12
        # player_stacks is a list of (x, y) of each stack
        player_stacks = X_stacks if player == X else O_stacks
        
        count = 0
        for pos in player_stacks:
            if pos in ring:
                count += 1
        return count/NORM

    def spacing(player):
        NORM = 1
        pass

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

    def control(player):
        ''' Blowing up all pieces now, how many squares are touched '''
        NORM = 1
        player_stacks = X_stacks if player == X else O_stacks
        
    def control2(player):
        ''' Moving and then blowing up, how many squares are touched '''
        NORM = 1
        pass

    def best_trade(player):
        ''' piece advantage of best trade '''
        NORM = 11
        pass

    def av_cluster_size(player):
        NORM = 1
        pass

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

    def column_piece_count(player, column):
        ''' How many pieces are in the column '''
        NORM = 24
        col = []
        for row in range(8):
            col.append(board[row][column])
        col = np.array(col)
        if player == X:
            return col[col > 0].sum()/NORM
        return -col[col < 0].sum()/NORM
        
    def column_stack_count(player, column):
        ''' How many stacks of certain size are in the column '''
        NORM = 8
        col = []
        for row in range(8):
            col.append(board[row][column])
        col = np.array(col)
        
        if player == X:
            return (col > 0).sum()/NORM
        return (col < 0).sum()/NORM
    
    def av_stack_size(player):
        NORM = 12
        return pieces(player) / stacks(player) / NORM
            
    # Distance to opponent
    # Measure of closeness or spread
    # Board position
    # Closeness to centre

    f1s = [largest_connected_cluster, #largest_almost_connected_cluster_stacks, largest_almost_connected_cluster_pieces,
           mobility, pieces, stacks, actions, connectivity, threat, av_stack_size]
    f2s = [piece_centrality, stack_centrality]
    f3s = [column_piece_count, column_stack_count]

    features = [f(player) for f in f1s for player in [X, O]] + \
               [f(player, ring) for f in f2s for ring in RINGS for player in [X, O]] + \
               [f(player, col) for f in f3s for col in range(8) for player in [X, O]]
    diffs = []
    for i in range(0, len(features, 2)):
        diffs.append(features[i] - features[i+1])
    features = np.array(features+diffs)
    
    memoized_states[state] = features
    return features

def main():
    print(Φ())

if __name__ == '__main__':
    main()