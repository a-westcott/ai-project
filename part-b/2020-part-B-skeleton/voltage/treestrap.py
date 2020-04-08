from player import State
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as dd

INF = 99.0
MAX_TURNS = 250

def Φ(state):
    X, O = 1, 0
    b = state.board

    def pieces(board, player):
        if player == X:
            return board[board > 0].sum()
        return -board[board < 0].sum()

    '''
    all_coords = [(x, y) for x in range(8) for y in range(8)]
        white_pieces = [(x, y) for x, y in all_coords if (self.board[x][y] == 0) or (self.board[x][y] > 0)]

        utility = 0
        utility +=  1  * self.board[self.board > 0].sum() / -self.board[self.board < 0].sum()

        # Distance to opponent

        # Measure of closeness or spread

        # Board position

        # Closeness to centre
        for x in range(8):
            for y in range(8):
                if self.board[x][y] > 0:
                    pass

        # Mobility
        utility += len(self.actions())/150
    '''
    return np.array([pieces(b, player) for player in [X, O]] + [pieces(b, X) - pieces(b, O)])

def H(s, θ):
    h = np.dot(Φ(s), θ)
    if h > 0.99*INF:
        return 0.99*INF
    if h < -0.99*INF:
        return -0.99*INF
    return h

α = 0.01
MAX_CHANGE = 0.1
def tree_strap_train(θ, depth=2):
    state = State()
    turn = 0
    random_turns = 0#np.random.choice([0] + [2]*2 + [4]*4 + [8]*8 + 16*[16] + 32*[32])
    
    history = dd(int)
    
    while (not state.terminal_test()) and (history[str(state)] < 4) and (turn < MAX_TURNS*2):
        print(f'Turn number {turn}')
        print(state)
        print()

        history[str(state)] += 1

        if turn < random_turns:
            state = state.result(state.actions()[np.random.choice([i for i in range(len(state.actions()))])])
        else:
            searched_states = []
            V = minimax(State(state.board), depth, θ, searched_states)
            searched_states.append((State(state.board), V, H(state, θ)))

            Δθ = np.zeros(3)
            for s, vs, hs in searched_states:
                #𝛿 = V(s) - H(s, θ)
                𝛿 = vs - hs
                Δθ += α*𝛿*Φ(s)
                s.board *= -1
                𝛿 = -(vs - hs)
                Δθ += α*𝛿*Φ(s)
            
            for i in range(len(Δθ)):
                if Δθ[i] > MAX_CHANGE:
                    Δθ[i] = MAX_CHANGE
                elif Δθ[i] < -MAX_CHANGE:
                    Δθ[i] = -MAX_CHANGE
            θ += Δθ

            actions = []
            for a in state.actions():
                child = state.result(a)
                actions.append((minimax(State(child.board*-1), depth-1, θ), a))
                
            state = state.result(max(actions)[1])
        
        state.board *= -1       
        turn += 1
    return θ

def minimax(state, depth, θ, searched_states=None):
    if state.terminal_test():
        return state.utility()
    if depth == 0:
        return H(state, θ)

    maxEval = -INF
    for a in state.actions():
        child = state.result(a)
        maxEval = max(maxEval, -minimax(State(-1*child.board), depth-1, θ, searched_states))
    
    if searched_states is not None:
        # Store the state, it's V(s) and H(s)
        searched_states.append((state, maxEval, H(state, θ)))
    return maxEval


N_GAMES = 1
def main():
    θ = np.array([-0.1, 0.1, -0.5])
    θs = [np.copy(θ)]
    for _ in range(N_GAMES):
        print('Game #', _)
        θ = tree_strap_train(θ)
        θs.append(θ)
        

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



if __name__ == '__main__':
    main()