import numpy as np
try:
    from state import State
    from player import ALL
except:
    from voltage_mkII.state import State
    from voltage_mkII.player import ALL

adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
buffer = [(0, 2), (2, 2), (1, 0), (-1, -2), (-2, -1), (-2, -2), (-1, -1), (-2, 1), (-1, 1), (0, -1), (0, -2), (0, 1), (2, -2), (2, -1), (1, 2), (2, 1), (-2, 0), (-1, 0), (-2, 2), (0, 0), (1, 1), (2, 0), (1, -2), (1, -1), (-1, 2)]

def n_v_one(state):
    '''
    Endgame move selector for being up n v 1 (n >= 2)
    '''
    X_stacks = [(x, y) for x, y in ALL if state.board[x][y] > 0]
    O_stack = [(x, y) for x, y in ALL if state.board[x][y] < 0][-1]
    Ox, Oy = O_stack

    # If we're all adjacent and opponent is adjacent, fix that 
    boomed = state.result(('BOOM', (Ox, Oy)))
    if not (boomed.board > 0).any() and not (boomed.board < 0).any():
        moves = sorted(state.actions(boom=False), reverse=True, key = lambda move: ((abs(Ox - move[3][0])) + (abs(Oy - move[3][1])))*move[1])
        return n_v_one_eval_moves(state, moves, Ox, Oy)

    # boom if opponent is adjacent to one of our pieces 
    # (we would have moved away above if this would take everything out)
    O_adj = [(Ox + x, Oy + y) for x, y in adj]
    for stack in X_stacks:
        if stack in O_adj:
            # boom that stack
            return ('BOOM', stack)

    # otherwise move closer
    moves = sorted(state.actions(boom=False), key = lambda move: ((abs(Ox - move[3][0]))**2 + (abs(Oy - move[3][1]))**2)/move[1])

    return n_v_one_eval_moves(state, moves, Ox, Oy)
        

def n_v_one_eval_moves(state, moves, Ox, Oy):
    '''
    Evaluate a list of moves ordered by preference, trying to ensure opponent can't destroy us 
    with a boom, and that we dont get into a fourth repeated state. If none can be found,
    try a repeated state one, otherwise it looks like we're not in good shape
    '''
    # try avoid repeated states and boom
    i=0
    while i < len(moves):
        potential = moves[i]
        result = state.result(potential)
        if result.history[result] >= 4:
            i += 1
            continue
        
        # check to make sure the oppenent cant boom us to a loss
        boomed = result.result(('BOOM', (Ox, Oy)))
        if not (boomed.board > 0).any() and not (boomed.board < 0).any():
            i += 1
            continue

        return potential
    
    # accept a draw
    i=0
    while i < len(moves):
        potential = moves[i]
        result = state.result(potential)
        if result.history[result] >= 4:
            return potential
    
    # accept a loss
    return potential[0]
