from player import State
N=8
import numpy as np
from treestrap import INF, negamax, minimax, num_features, H
from collections import defaultdict as dd
from features import Φ, ALL_STACKS, RINGS

TRAIN_DEPTH = 2


def translate_actions(actions, verbose=False):
    readable_actions = []
    for action in actions:
        if action[0] == 'BOOM':
            if verbose:
                readable_actions.append(('BOOM', translate_coord(action[1])))
            else:
                readable_actions.append((translate_coord(action[1]),))
        else:
            if verbose:
                readable_actions.append(('MOVE', action[1], translate_coord(action[2]), translate_coord(action[3])))
            else:
                readable_actions.append((action[1], translate_coord(action[2]), translate_coord(action[3])))
    
    return (readable_actions)

def translate_coord(coord):
    x,y = coord
    return 8*x + y

def actions_with_indices(actions):
    return [(str(i) + ':', actions[i]) for i in range(len(actions))]

def print_board(board):
    d = {k: v for k, v in zip(list(range(13)), [str(i) for i in range(10)] + ['A', 'B', 'C'])}
    board = [[d[abs(pos)] if pos >= 0 else '-'+d[abs(pos)] for pos in row] for row in board]
    print(25*'_' + '\n|' + '\n|'.join([['|'.join([str(pos).rjust(2) for pos in row]) for row in board][k] + '| ' +
           [''.join([str(i+8*j).rjust(3) for i in range(8)]) for j in range(8)][k] for k in range(8)]))

α = 0.001
λ = 0.5
MAX_CHANGE = 0.1
def play(θo, θm, θe, depth=TRAIN_DEPTH):
    OPN, MID, END = 0, 1, 2
    state = State()

    first = np.random.choice([0, 1])

    random_turns = 0#np.random.choice([0] + [2]*2 + [4]*4 + [8]*8 + 16*[16] + 32*[32])
    while (not state.terminal_test()):
        print(f'Turn number {state.turn}')
        print_board(state.board)
        print()

        if (state.turn+first)%2:
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
                
                for i in range(num_features):
                    if Δθ[i] > MAX_CHANGE:
                        Δθ[i] = MAX_CHANGE
                    elif Δθ[i] < -MAX_CHANGE:
                        Δθ[i] = -MAX_CHANGE
                θ += Δθ

                actions = []
                actions2 = []
                for a in state.actions():
                    child = state.result(a)
                    actions.append((-negamax(State(-1*child.board), -INF, INF, depth-1, θ), a))
                    
                state = state.result(max(actions)[1])
        else:
            print(actions_with_indices(translate_actions(state.actions())))
            i = int(input())
            state = state.result(state.actions()[i])

        state.board *= -1
        state.turn += 1
    print(state)
    print('Game over!')
    return θo, θm, θe

def main():
    θo = np.random.uniform(-0.01, 0.01, num_features)
    θm = np.random.uniform(-0.01, 0.01, num_features)
    θe = np.random.uniform(-0.01, 0.01, num_features)
    
    play(θo, θm, θe, depth=TRAIN_DEPTH, tdl=False)

    
if __name__ == '__main__':
    main()