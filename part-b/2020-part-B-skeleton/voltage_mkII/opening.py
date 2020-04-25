try:
    from state import State, MOVE, BOOM
    location1 = 'book1.pkl'
    location2 = 'book2.pkl'
except:
    from voltage_mkII.state import State, MOVE, BOOM
    location1 = 'voltage_mkII/book1.pkl'
    location2 = 'voltage_mkII/book2.pkl'


import numpy as np
import pickle

with open(location1, 'rb') as f:
    opening_book = pickle.load(f)

with open(location2, 'rb') as f:
    opening_book2 = pickle.load(f)

opening_book.update(opening_book2)
del opening_book2

# opening_book[state.board.tostring()] = move

def opening_move(state, turn):
    LEFT, RGHT = 0, 1
    side = LEFT if np.sign(state.board[0][0] + state.board[7][0]) > 0 else RGHT
    if turn < 2:
        return [MOVE, 1, (4, 0), (4, 1)] if side == LEFT else [MOVE, 1, (4, 7), (4, 6)]
    if turn < 4:
        return [MOVE, 2, (4, 1), (4, 3)] if side == LEFT else [MOVE, 2, (4, 6), (4, 4)]
    # White
    if turn == 4:
        if state.board[4][5]:
            return [MOVE, 2, (4, 3), (4, 4)]
        else:        
            return [MOVE, 2, (4, 3), (4, 5)]
    # Black
    if turn == 5:
        if side == LEFT:
            if state.board[4][4] < 0 and state.board[4][5] < 0:
                return [BOOM, (4, 3)]
            elif state.board[4][5] == 0:
                return [MOVE, 2, (4, 3), (4, 5)]
            elif state.board[4][4] == 0:
                return [MOVE, 2, (4, 3), (4, 4)]
            else:
                print(state)
                raise ValueError("Something happened that Cameron didn't forsee")
        else:
            if state.board[4][3] < 0 and state.board[4][2] < 0:
                return [BOOM, (4, 4)]
            elif state.board[4][2] == 0:
                return [MOVE, 2, (4, 4), (4, 2)]
            elif state.board[4][3] == 0:
                return [MOVE, 2, (4, 4), (4, 3)]
            else:
                print(state)
                raise ValueError("Something happened that Cameron didn't forsee")
                
    if turn == 6:
        if side == LEFT:
            # If opponent tried to block
            if state.board[4][4] > 0:
                # If opponent has moved away
                if state.board[3][4] == 0 and state.board[3][5] == 0 and state.board[3][6] == 0 and \
                   state.board[4][6] == 0 and \
                   state.board[5][4] == 0 and state.board[5][5] == 0 and state.board[5][6] == 0:
                    return [MOVE, 2, (4, 4), (4, 6)]
                elif state.board[4][5] < 0 and (state.board[3][5] < 0 or state.board[5][5] < 0):
                    return [BOOM, (4, 4)]
                elif state.board[4][5] < 0:
                    return [MOVE, 2, (4, 4), (4, 5)]
            else:        
                # If opponent played optimally
                if sorted([state.board[3][6], state.board[3][7], state.board[4][6], state.board[4][7]]) == [-2, 0, 0, 0]:
                    # If opponent has moved to inbetween cluster regions, move to that side
                    if state.board[5][6:].sum() < 0:
                        if state.board[5][6] < 0:
                            return [BOOM, (4, 5)]
                        else:
                            return [MOVE, 1, (4, 5), (6, 5)]
                    elif state.board[2][6:].sum() < 0:
                        if state.board[2][6] < 0:
                            return [MOVE, 1, (4, 5), (2, 5)]
                        else:
                            return [MOVE, 1, (4, 5), (6, 5)]

                # If just 4 down below, go there
                if state.board[6][6] == -1 and state.board[6][7] == -1 and state.board[7][6] == -1 and state.board[7][7] == -1:
                    if state.board[6][5] == 0:
                        return [MOVE, 2, (4, 5), (6, 5)]
                    else:
                        return [MOVE, 1, (4, 5), (5, 5)]
                # 4 in middle, move up
                elif state.board[3][6] == -1 and state.board[3][7] == -1 and state.board[4][6] == -1 and state.board[4][7] == -1:
                    # If vulnerable below
                    if (sorted([state.board[6][6], state.board[7][6]]) in [[-1, -1], [-2, -1], [-2, -2]]) or \
                       (sorted([state.board[6][6], state.board[7][6]]) == [-1, 0] and (state.board[6][7] < 0 or state.board[7][7])):
                        if state.board[6][5] == 0:
                            return [MOVE, 1, (4, 5), (6, 5)]
                        else:
                            return [MOVE, 1, (4, 5), (5, 5)]
                    elif state.board[2][5] == 0:
                        return [MOVE, 1, (4, 5), (2, 5)]
                    else:
                        return [MOVE, 1, (4, 5), (5, 5)]
                # 4 above
                else:
                    return [BOOM, (4, 5)]
        else:
             # If opponent tried to block
            if state.board[4][3] > 0:
                # If opponent has moved away
                if state.board[3][3] == 0 and state.board[3][2] == 0 and state.board[3][1] == 0 and \
                   state.board[4][1] == 0 and \
                   state.board[5][3] == 0 and state.board[5][2] == 0 and state.board[5][1] == 0:
                    return [MOVE, 2, (4, 3), (4, 1)]
                elif state.board[4][2] < 0 and (state.board[3][2] < 0 or state.board[5][2] < 0):
                    return [BOOM, (4, 3)]
                elif state.board[4][2] < 0:
                    return [MOVE, 2, (4, 3), (4, 2)]
            else:        
                # If opponent played optimally
                if sorted([state.board[3][1], state.board[3][0], state.board[4][1], state.board[4][0]]) == [-2, 0, 0, 0]:
                    # If opponent has moved to inbetween cluster regions, move to that side
                    if state.board[5][:2].sum() < 0:
                        if state.board[5][1] < 0:
                            return [BOOM, (4, 2)]
                        else:
                            return [MOVE, 1, (4, 2), (6, 2)]
                    elif state.board[2][:2].sum() < 0:
                        if state.board[2][1] < 0:
                            return [MOVE, 1, (4, 2), (2, 2)]
                        else:
                            return [MOVE, 1, (4, 2), (6, 2)]

                # If just 4 down below, go there
                if state.board[6][1] == -1 and state.board[6][0] == -1 and state.board[7][1] == -1 and state.board[7][0] == -1:
                    if state.board[6][2] == 0:
                        return [MOVE, 2, (4, 2), (6, 2)]
                    else:
                        return [MOVE, 1, (4, 2), (5, 2)]
                # 4 in middle, move up
                elif state.board[3][1] == -1 and state.board[3][0] == -1 and \
                     state.board[4][1] == -1 and state.board[4][0] == -1:
                    # If vulnerable below
                    if (sorted([state.board[6][1], state.board[7][1]]) in [[-1, -1], [-2, -1], [-2, -2]]) or \
                       (sorted([state.board[6][1], state.board[7][1]]) == [-1, 0] and (state.board[6][0] < 0 or 
                                                                                       state.board[7][0])):
                        if state.board[6][2] == 0:
                            return [MOVE, 1, (4, 2), (6, 2)]
                        else:
                            return [MOVE, 1, (4, 2), (5, 2)]
                    elif state.board[2][2] == 0:
                        return [MOVE, 1, (4, 2), (2, 2)]
                    else:
                        return [MOVE, 1, (4, 2), (5, 2)]
                # 4 above
                else:
                    return [BOOM, (4, 2)]





