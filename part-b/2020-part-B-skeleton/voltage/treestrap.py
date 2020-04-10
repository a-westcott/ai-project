from player import State, ALL, MOVE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as dd

INF = 99.0
TRAIN_DEPTH = 2
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

def Φ(state):
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
        player_stacks = X_stacks.copy() if player == X else O_stacks.copy()
        colour = 1 if player == X else -1
        adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
        
        largest_connected_cluster = 0
        
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

        return largest_connected_cluster

    def largest_almost_connected_cluster(player):
        ''' 
        number of pieces in an extended cluster
        (vulnerable to one opposing stack in the right spot)
        this is probably hard to program
        '''
        pass

    def piece_centrality(player, ring):
        ''' Overall centrality in terms of where pieces are '''
        # player_stacks is a list of (x, y) of each stack
        player_stacks = X_stacks if player == X else O_stacks
        colour = 1 if player == X else -1
        
        count = 0
        for pos in player_stacks:
            if pos in ring:
                count += board[pos[0]][pos[1]]
        return count*colour

    def stack_centrality(player, ring, stack_size=ALL_STACKS):
        ''' Overall centrality in terms of where stacks are (ignores that bigger stacks have more pieces) '''
        # player_stacks is a list of (x, y) of each stack
        if stack_size == ALL_STACKS:
            player_stacks = X_stacks if player == X else O_stacks
        else:
            player_stacks = X_stacks_by_size[stack_size-1] if player == X else O_stacks_by_size[stack_size-1]
        
        count = 0
        for pos in player_stacks:
            if pos in ring:
                count += 1
        return count

    def spacing(player):
        pass

    def mobility(player, stacksize=ALL_STACKS):
        height = abs(board[x][y])

        # all possible move positions on the board, including moves on to opposing pieces
        all_coords = []
        for dx in range(1, height+1):
            if 0 <= x+dx < 8:
                all_coords.append((x+dx, y))
            if 0 <= x-dx < 8:
                all_coords.append((x-dx, y))

        for dy in range(1, height+1):
            if 0 <= y+dy < 8:
                all_coords.append((x, y+dy))
            if 0 <= y-dy < 8:
                all_coords.append((x, y-dy))

        # make sure we dont move onto an opposing colour
        all_coords = [(x1, y1) for x1, y1 in all_coords if (board[x1][y1] == 0) or (board[x1][y1] > 0)]

        # append all the moves with all possible number of pieces moved
        for n in range(1, height+1):
            for x_dest, y_dest in all_coords:
                actions.append([MOVE, n, (x, y), (x_dest, y_dest)])

    def control(player):
        ''' Blowing up all pieces now, how many squares are touched '''
        player_stacks = X_stacks if player == X else O_stacks
        

    def control2(player):
        ''' Moving and then blowing up, how many squares are touched '''
        pass

    def best_trade(player):
        ''' piece advantage of best trade '''
        pass

    def av_cluster_size(player):
        pass

    

    def pieces(player):
        if player == X:
            return board[board > 0].sum()
        return -board[board < 0].sum()
    
    def stacks(player, stack_size=ALL_STACKS):
        if stack_size == ALL_STACKS:
            if player == X:
                return (board > 0).sum()
            return (board < 0).sum()
            
        if player == X:
            return (board == stack_size).sum()
        return (board == -stack_size).sum()
    
    def actions(player):
        if player == X:
            return len(State(board).actions())
        return len(State(opp_b).actions())
    
    def connectivity(player):
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
        return len(s)
            
    def threat(player):
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
        return len(s)
    
    def av_stack_size(player):
        return pieces(player) / stacks(player)
            
    # Distance to opponent
    # Measure of closeness or spread
    # Board position
    # Closeness to centre

    functions = [pieces, stacks, actions, connectivity, threat, av_stack_size]
    return np.array([f(player) for f in functions for player in [X, O]] +
                    [f(X) - f(O) for f in functions] + 
                    [])

def H(features, θ):
    h = np.dot(features, θ)
    if h > 0.99*INF:
        return 0.99*INF
    if h < -0.99*INF:
        return -0.99*INF
    return h

α = 0.0001
λ = 0.5
MAX_CHANGE = 0.1
def tree_strap_train(θ, depth=TRAIN_DEPTH):
    state = State()
    random_turns = np.random.choice([0] + [2]*2 + [4]*4 + [8]*8 + 16*[16] + 32*[32])
    while (not state.terminal_test()):
        print(f'Turn number {state.turn}')
        print(state)
        print()


        state.history[state] += 1

        if state.turn < random_turns:
            num_actions = len(state.actions(False))
            state = state.result(state.actions(False)[np.random.choice([i for i in range(num_actions)])])
        else:
            searched_states = []
            V = minimax(State(state.board), depth, θ, searched_states)

            Δθ = np.zeros(num_features)
            for s, vs, hs, features, d in searched_states:
                #𝛿 = V(s) - H(features, θ)
                𝛿 = vs - hs
                Δθ += α*𝛿*features*λ**(depth-d)
                s.board *= -1
                𝛿 = -(vs - hs)
                Δθ += α*𝛿*Φ(s)*λ**(depth-d)
            
            for i in range(num_features):
                if Δθ[i] > MAX_CHANGE:
                    Δθ[i] = MAX_CHANGE
                elif Δθ[i] < -MAX_CHANGE:
                    Δθ[i] = -MAX_CHANGE
            θ += Δθ

            actions = []
            for a in state.actions():
                child = state.result(a)
                actions.append((-minimax(State(child.board*-1), depth-1, θ), a))
                
            state = state.result(max(actions)[1])
        
        state.board *= -1       
        state.turn += 1
    return θ

def minimax(state, depth, θ, searched_states=None):
    if state.terminal_test():
        return state.utility()
    if depth == 0:
        return H(Φ(state), θ)

    maxEval = -INF
    for a in state.actions():
        child = state.result(a)
        maxEval = max(maxEval, -minimax(State(-1*child.board), depth-1, θ, searched_states))
    
    if searched_states is not None:
        # Store the state, it's V(s) and H(s)
        features = Φ(state)
        searched_states.append((state, maxEval, H(features, θ), features, depth))
    return maxEval


N_GAMES = 5
def main():
    θ = np.array([-0.31923688,  0.25083284, -0.30244211, -0.0489124,  -0.3328332,   0.09468402,
  0.48055289,  0.47096562,  0.07606004,  0.04764147, -0.37635794, -0.09468402,
 -0.16457624,  0.29515123,  0.4352868,  -0.73503295, -0.01776477, -0.24063195])
    θs = [np.copy(θ)]
    for _ in range(N_GAMES):
        print('Game #', _)
        θ = tree_strap_train(θ)
        θs.append(np.copy(θ))
        

    fig, ax = plt.subplots(figsize=(20, 10))
    for i in range(len(θ)):
        x = [j for j in range(len(θs))]
        y = [θs[j][i] for j in range(len(θs))]
        ax.plot(x, y)
    sns.despine()
    plt.savefig('weights.png')

    FACTOR=3
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    lbls = ['#X', '#O', 'XAdv']

    fig, ax = plt.subplots(figsize=(20, 2))
    sns.heatmap([θ], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, ax=ax, xticklabels=lbls)
    plt.savefig('weight.png')

    print(θ)

num_features = len(Φ(State()))
if __name__ == '__main__':
    main()