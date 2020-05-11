from player import State, ALL, MOVE, INF, OPN, DEV, MID, END
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict as dd
from features import Φ, ALL_STACKS, RINGS, H
from ab_treestrap_train import alpha_beta_train, ab_weight_updates
from opening import opening_book
#from weight import weight1

from multiprocessing import Pool, Manager
import gc

MULTI = False
PROCESSES = 8

AB_TRAIN = True
TRAIN_DEPTH = 4


num_features = len(Φ(State(), {}))

α = 0.000001
λ = 0.5
MAX_CHANGE = 0.01
def tree_strap_train(θo, θd, θm, θe, depth=TRAIN_DEPTH):
    state = State()
    #memoised_features = {} if MULTI else None

    memoised_features = {}
    random_turns = np.random.choice([0]*0 + [2]*0 + [6]*2 + [8]*4 + [16]*4 + [32]*8)
    # See if each player will use book    
    X_use_book = np.random.choice([0, 0, 0, 1])
    O_use_book = np.random.choice([0, 0, 0, 1])

    while (not state.training_terminal_test()):
        print(f'Turn number {state.turn}')
        print(state)
        print()
        if state.stage[0] == OPN:
            θ = θo
        elif state.stage[0] == DEV:
            θ = θd
        elif state.stage[0] == MID:
            θ = θm
        else:
            θ = θe
            #depth = 2*TRAIN_DEPTH


        if ((state.turn%2 and X_use_book) or (not state.turn%2 and O_use_book)) and (str(state.board) in opening_book):
            state = state.result(tuple(opening_book[str(state.board)]))

        elif state.turn < random_turns:
            num_actions = len(state.actions(False))
            state = state.result(state.actions(False)[np.random.choice([i for i in range(num_actions)])])
        else:
            if MULTI:
                searched_states = set()
                V = speedy_minimax(state, depth, θ, searched_states, first=True, memoised_states=memoised_features)[0]
            elif not AB_TRAIN:
                searched_states = []
                V = negamax(state, -10*INF, 10*INF, depth, θ, memoised_features)
            
            if AB_TRAIN:
                searched_states = []
                alpha_beta_train(state, θ, searched_states, TRAIN_DEPTH, memoised_features)
                ab_weight_updates(searched_states, θ, depth, α, λ, MAX_CHANGE)
            else:
                Δθ = np.zeros(num_features)
                #for s, vs, hs, features, d in searched_states:
                #    # updates should only happen for states that match the player to play
                #    if not d % 2:
                #        features = np.frombuffer(features)
                #        #𝛿 = V(s) - H(features, θ)
                #        𝛿 = vs - hs
                #        Δθ += α*𝛿*features*λ**(depth-d)
                if V != 0:
                    features = Φ(state, memoised_features)
                    h = H(features, θ)
                    𝛿 = V - h          
                    Δθ += α*𝛿*features  

                for i in range(num_features):
                    if Δθ[i] > MAX_CHANGE:
                        Δθ[i] = MAX_CHANGE
                    elif Δθ[i] < -MAX_CHANGE:
                        Δθ[i] = -MAX_CHANGE
                θ += Δθ

            best_action = None
            alpha, beta, v = -4*INF, 4*INF, -4*INF
            for a in state.actions():
                child = state.result(a)
                nmax = -negamax(child, -beta, -alpha, depth-1, θ, memoised_features)
                if nmax > alpha:
                    alpha = nmax
                    best_action = a

    

            
            state = state.result(best_action)
            print(alpha)
    
    print('Terminal State:')
    print(state)
    memoised_features = None
    gc.collect()
    return θo, θd, θm, θe

def minimax(state, depth, θ, searched_states=None):
    if state.stages_terminal_test():
        return state.utility()
    if depth == 0:
        return H(Φ(state), θ)

    maxEval = -INF
    for a in state.actions():
        child = state.result(a)
        #child.board *= -1
        maxEval = max(maxEval, -minimax(child, depth-1, θ, searched_states))
    
    if searched_states is not None:
        # Store the state, it's V(s) and H(s)
        features = Φ(state)
        searched_states.append((state, maxEval, H(features, θ), features, depth))
    return maxEval

def speedy_minimax(state, depth, θ, searched_states=None, first=False, memoised_states=None):
    if state.training_terminal_test():
        return state.utility(), searched_states
    if depth == 0:
        return H(Φ(state, memoised_states), θ), searched_states

    maxEval = -INF
    # set up multiprocessing sharing the memoised states dict, 
    if first:
        with Manager() as m:
            d = m.dict(memoised_states)
            child = state.result(a)
            #child.board *= -1
            children = [(child, depth-1, θ, searched_states, False, d) for a in state.actions()]
            with Pool(PROCESSES) as p:
                results = (p.starmap(speedy_minimax, children))
            memoised_states.update(dict(d))

        evals = [res[0] for res in results]
        maxEval = max(evals)
        sets = [res[1] for res in results]
        for s in sets:
            searched_states.update(s)
    else:
        for a in state.actions():
            child = state.result(a)
            #child.board *= -1
            maxEval = max(maxEval, -speedy_minimax(child, depth-1, θ, searched_states, memoised_states=memoised_states)[0])
        
    if searched_states is not None:
        # Store the state, it's V(s) and H(s)
        features = Φ(state, memoised_states)
        searched_states.add((state.__hash__(), maxEval, H(features, θ), features.tostring(), depth))
    return maxEval, searched_states

def negamax(state, alpha, beta, depth, θ, memoised_states=None):
    if state.training_terminal_test():
        return state.utility(train=True)
    if depth == 0:
        if memoised_states:
            return H(Φ(state, memoised_states), θ)
        return H(Φ(state), θ)

    v = -4*INF
    for a in state.actions():
        child = state.result(a)
        v = max(v, -negamax(child, -beta, -alpha, depth-1, θ, memoised_states))
        if v >= beta:
            return v
        alpha = max(alpha, v)        
    return v

N_GAMES = 50000
def main():
    try:
        θo = np.load('w_opn-ab3.npy')
        θd = np.load('w_dev-ab3.npy')
        θm = np.load('w_mid-ab3.npy')
        θe = np.load('w_end-ab3.npy')
        #θo = np.load('o33.npy')
        #θd = np.load('d33.npy')
        #θm = np.load('m33.npy')
        #θe = np.load('e33.npy')

    except:
        θo = np.copy(weight1)
        θd = np.copy(θo)
        θm = np.copy(θo)
        θe = np.copy(θo)

    θos, θds, θms, θes = [np.copy(θo)], [np.copy(θd)], [np.copy(θm)], [np.copy(θe)]
    for game_num in range(N_GAMES):
        print('Game #', game_num+1)
        θo, θd, θm, θe = tree_strap_train(θo, θd, θm, θe, depth=TRAIN_DEPTH)
        θos.append(np.copy(θo))
        θds.append(np.copy(θd))
        θms.append(np.copy(θm))
        θes.append(np.copy(θe))
        np.save('w_opn', θo)
        np.save('w_dev', θd)
        np.save('w_mid', θm)
        np.save('w_end', θe)
        
        gc.collect()

        if game_num%10 == 0:

            fig = plt.figure(figsize=(20, 20))
            plt.subplot(4, 1, 1)
            for i in range(len(θo)):
                x = [j for j in range(len(θos))]
                y = [θos[j][i] for j in range(len(θos))]
                plt.plot(x, y)
                plt.title('Opening', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 2)
            for i in range(len(θm)):
                x = [j for j in range(len(θms))]
                y = [θms[j][i] for j in range(len(θms))]
                plt.plot(x, y)
                plt.title('Mid-Game', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 3)
            for i in range(len(θm)):
                x = [j for j in range(len(θms))]
                y = [θms[j][i] for j in range(len(θms))]
                plt.plot(x, y)
                plt.title('Mid-Game', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 4)
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
            plt.subplot(4, 1, 1); sns.heatmap([θo], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
            plt.subplot(4, 1, 2); sns.heatmap([θd], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
            plt.subplot(4, 1, 3); sns.heatmap([θm], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
            plt.subplot(4, 1, 4); sns.heatmap([θe], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, xticklabels=[])
            plt.savefig('Unlabelled-Heatmap.png')
            
            fig = plt.figure(figsize=(20, 20))
            plt.subplot(4, 1, 1)
            for i in range(len(θo)):
                x = [j for j in range(len(θos))]
                y = [θos[j][i] for j in range(len(θos))]
                plt.plot(x, y)
                plt.title('Opening', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 2)
            for i in range(len(θd)):
                x = [j for j in range(len(θds))]
                y = [θds[j][i] for j in range(len(θds))]
                plt.plot(x, y)
                plt.title('Develop', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 3)
            for i in range(len(θm)):
                x = [j for j in range(len(θms))]
                y = [θms[j][i] for j in range(len(θms))]
                plt.plot(x, y)
                plt.title('Mid-Game', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
            sns.despine()

            plt.subplot(4, 1, 4)
            for i in range(len(θe)):
                x = [j for j in range(len(θes))]
                y = [θes[j][i] for j in range(len(θes))]
                plt.plot(x, y)
                plt.title('Ending', fontsize=16)
                plt.ylabel('Weight', fontsize=14)
                plt.xlabel('Game', fontsize=14)
            sns.despine()
            plt.savefig('Training.png')

            '''
            f1s = ['largest_connected_cluster', 'mobility', 'pieces', 'stacks', 'actions', 'connectivity', 'threat', 'av_stack_size']
            f2s = ['piece_centrality', 'stack_centrality']
            f3s = ['column_piece_count', 'column_stack_count']

            lbls = [f+player for f in f1s for player in ['X', 'O']] +\
                [f+str(ring)+player for f in f2s for ring in range(4) for player in ['X', 'O']]

            diffs = []
            for i in range(0, len(lbls), 2):
                f = lbls[i][:-1]
                diffs.append(f+'_diff')
            lbls += diffs
            
            fig = plt.figure(figsize=(4, 40))
            plt.subplot(1, 4, 1); sns.heatmap([[e] for e in θo], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=lbls, xticklabels=[])
            plt.subplot(1, 4, 2); sns.heatmap([[e] for e in θd], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=[], xticklabels=[])
            plt.subplot(1, 4, 3); sns.heatmap([[e] for e in θm], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=[], xticklabels=[])
            plt.subplot(1, 4, 4); sns.heatmap([[e] for e in θe], cmap=cmap, vmin=-FACTOR, vmax=FACTOR, yticklabels=[], xticklabels=[])
            plt.savefig('Labelled-Heatmap.png')
            '''
            plt.close('all')

    print(θo)
    print(θd)
    print(θm)
    print(θe) 

if __name__ == '__main__':
    main()