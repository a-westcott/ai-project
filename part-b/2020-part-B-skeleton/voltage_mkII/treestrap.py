from player import State, ALL, MOVE, INF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as dd
from features import Φ, ALL_STACKS, RINGS

TRAIN_DEPTH = 2

num_features = len(Φ(State()))

def H(features, θ):
    h = np.dot(features, θ)
    if h > 0.99*INF:
        return 0.99*INF
    if h < -0.99*INF:
        return -0.99*INF
    return h

α = 0.000001*3
λ = 0.5
MAX_CHANGE = 0.1
def tree_strap_train(θo, θm, θe, depth=TRAIN_DEPTH):
    OPN, MID, END = 0, 1, 2
    state = State()
    random_turns = np.random.choice([0] + [2]*2 + [4]*4 + [8]*8 + 16*[16] + 32*[32])
    while (not state.terminal_test()):
        print(f'Turn number {state.turn}')
        print(state)
        print()

        if state.board[state.board > 0].sum() == 12:
            θ = θo
        elif state.board[state.board > 0].sum() > 5: 
            θ = θm
        else:
            θ = θe

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
                #s.board *= -1
                #flipped_features = Φ(s)
                #𝛿 = -(vs - hs) THIS IS ALL WRONG BTW, RECALCULATE V AND H
                #Δθ += α*𝛿*flipped_features*λ**(depth-d)
            
            for i in range(num_features):
                if Δθ[i] > MAX_CHANGE:
                    Δθ[i] = MAX_CHANGE
                elif Δθ[i] < -MAX_CHANGE:
                    Δθ[i] = -MAX_CHANGE
            θ += Δθ

            actions = []
            for a in state.actions():
                child = state.result(a)
                actions.append((-negamax(State(-1*child.board), -INF, INF, depth-1, θ), a))
                
            state = state.result(max(actions)[1])

        state.board *= -1
        state.turn += 1
    return θo, θm, θe

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


def negamax(state, alpha, beta, depth, θ):
    if state.terminal_test():
        return state.utility()
    if depth == 0:
        return H(Φ(state), θ)

    v = -INF
    for a in state.actions():
        child = state.result(a)
        # game state must be flipped
        v = max(v, -negamax(State(-1*child.board), alpha, beta, depth-1, θ))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v

N_GAMES = 1
def main():
    θo = np.load('opening.npy')
    θm = np.load('middle.npy')
    θe = np.load('end.npy')

    θos, θms, θes = [np.copy(θo)], [np.copy(θm)], [np.copy(θe)]
    for _ in range(N_GAMES):
        print('Game #', _)
        θo, θm, θe = tree_strap_train(θo, θm, θe, depth=TRAIN_DEPTH)
        θos.append(np.copy(θo))
        θms.append(np.copy(θm))
        θes.append(np.copy(θe))
        np.save('opening', θo)
        np.save('middle', θm)
        np.save('end', θe)
        if not _%10:
            fig = plt.figure(figsize=(20, 20))
            plt.subplot(3, 1, 1)
            for i in range(len(θo)):
                x = [j for j in range(len(θos))]
                y = [θos[j][i] for j in range(len(θos))]
                plt.plot(x, y)
                plt.title('Opening', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(3, 1, 2)
            for i in range(len(θm)):
                x = [j for j in range(len(θms))]
                y = [θms[j][i] for j in range(len(θms))]
                plt.plot(x, y)
                plt.title('Mid-Game', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(3, 1, 3)
            for i in range(len(θe)):
                x = [j for j in range(len(θes))]
                y = [θes[j][i] for j in range(len(θes))]
                plt.plot(x, y)
                plt.title('Ending', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
                plt.xlabel('Game', fontsize=14)
            sns.despine()
            plt.savefig('Training.png')

        
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    fig = plt.figure(figsize=(20, 4))
    FACTOR=1
    plt.subplot(3, 1, 1); sns.heatmap([θo], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
    plt.subplot(3, 1, 2); sns.heatmap([θm], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
    plt.subplot(3, 1, 3); sns.heatmap([θe], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
    plt.savefig('Unlabelled-Heatmap.png')
    

    fig = plt.figure(figsize=(20, 20))
    plt.subplot(3, 1, 1)
    for i in range(len(θo)):
        x = [j for j in range(len(θos))]
        y = [θos[j][i] for j in range(len(θos))]
        plt.plot(x, y)
        plt.title('Opening', fontsize=16)
        plt.ylabel('Weight', fontsize=14)
    sns.despine()

    plt.subplot(3, 1, 2)
    for i in range(len(θm)):
        x = [j for j in range(len(θms))]
        y = [θms[j][i] for j in range(len(θms))]
        plt.plot(x, y)
        plt.title('Mid-Game', fontsize=16)
        plt.ylabel('Weight', fontsize=14)
    sns.despine()

    plt.subplot(3, 1, 3)
    for i in range(len(θe)):
        x = [j for j in range(len(θes))]
        y = [θes[j][i] for j in range(len(θes))]
        plt.plot(x, y)
        plt.title('Ending', fontsize=16)
        plt.ylabel('Weight', fontsize=14)
        plt.xlabel('Game', fontsize=14)
    sns.despine()
    plt.savefig('Training.png')

    lbls = []
    fig = plt.figure(figsize=(4, 40))
    plt.subplot(1, 3, 1); sns.heatmap([[e] for e in θo], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=lbls, xticklabels=[])
    plt.subplot(1, 3, 2); sns.heatmap([[e] for e in θm], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=[], xticklabels=[])
    plt.subplot(1, 3, 3); sns.heatmap([[e] for e in θe], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=[], xticklabels=[])
    plt.savefig('Labelled-Heatmap.png')
    
    print(θo)
    print(θm)
    print(θe) 

if __name__ == '__main__':
    main()