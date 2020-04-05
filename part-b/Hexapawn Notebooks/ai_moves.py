from hexapawn import State, INF, X, O, BLANK, N, DOWN, UP 

def ab_pruning_move(state, depth=1, j=False):
    if depth is None:
        raise ValueError 
    lst = []
    
    for a in state.actions():
        result = State(state.turn, state)
        result.move(a)
        # If terminal state, no need to search
        if result.terminal_test() is not None:
            rn = 1 if result.terminal_test() == X else -1
            lst.append((rn, a))
        else:
            isMax = True if result.turn == X else False
            lst.append((ab_pruning(result, depth, -INF, INF, isMax=isMax, j=j), a))
    
    if state.turn == X:
        return max(lst)[1]
    return min(lst)[1]

def TDLeaf_move(state, depth=1):
    if depth is None:
        raise ValueError 
    return ab_pruning_move(state, depth, j=True)
    
def TD_move(state, depth=None):
    if depth is not None:
        raise ValueError 
    lst = []
    
    for a in state.actions():
        result = State(state.turn, state)
        result.move(a)
        # If terminal state, no need to search
        if result.terminal_test() is not None:
            rn = 1 if result.terminal_test() == X else -1
            lst.append((rn, a))
        else:
            lst.append((J(result.board, w), a))
    
    if player == X:
        return max(lst)[1] 
    return min(lst)[1]
    
    
def random_move(state, depth=None):
    if depth is not None:
        raise ValueError 
    actions = state.actions()
    return actions[np.random_choice([i for i in range(len(actions))])]
    
    
def ab_pruning(state, depth, α, β, isMax, j=False):
    if state.terminal_test() is not None:
        return state.utility()
    if depth == 0:
        if j:
            return J(state.board, w)
        return state.utility()
    
    if isMax:
        maxV = -INF
        for a in state.actions():
            result = State(state.turn, state)
            result.move(a)
            value = ab_pruning(result, depth-1, α, β, isMax=False, j=j)
            maxV = max(maxV, value)
            α = max(α, value)
            if β <= α:
                break
        return maxV
    else:
        minV = +INF
        for a in state.actions():
            result = State(state.turn, state)
            result.move(a)
            value = ab_pruning(result, depth-1, α, β, isMax=True, j=j)
            minV = min(minV, value)
            β = min(β, value)
            if β <= α:
                break
        return minV
    
  