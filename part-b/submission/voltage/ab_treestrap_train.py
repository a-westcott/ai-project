U_BOUND, EXACT, L_BOUND = 1, 0, -1

import numpy as np

from features import H, Φ,INF, num_features 
from state import State


def alpha_beta_train(state, θ, searched_states, depth, memoised_features=None):
    '''
    Assume always called with max first, even depth
    return value takes form:
    max_d0_eval, max_d1_eval

    ie the best depth 0 state from max's perspective, 
    and the best depth 1 state form mins perspective (ignorant of the layer below)

    We're basically doing two searches, an alpha beta for max, 
    and a sort of minimax (?) on the tree that max expores for min. 

    Upper and lower bounds only matter for being added into searched_states;
    if we have an upper/lower bound, that value wont be the one being 
    used further up the tree, and only potentially for weight updates.
    I think this is true, it almost certainly is for max, and im fairly
    confident it is for min (maybe less confident if depth > 4...)
    '''
    assert(depth %2 != 1)
    a = max_value(state, -4*INF, 4*INF, depth, θ, searched_states, memoised_features)
    return a

def max_value(state, alpha, beta, depth, θ, searched_states, memoised_features=None):
    if state.training_terminal_test():
        return state.utility(train=True), -state.utility(train=True)
    if depth == 0:
        v0 = H(Φ(state, memoised_features), θ)
        return v0, -INF*4

    v0, v1 = -4*INF, 4*INF
    for a in state.actions():
        child = state.result(a)
        pot_v0, pot_v1 = min_value(child, alpha, beta, depth-1, θ, searched_states, memoised_features)
        v1 = min(v1, pot_v1)
        v0 = max(v0, pot_v0)
        if v0 >= beta:
            if depth > 1:
                # update searched states
                # v0 is a L_BOUND, v1 is a U_BOUND
                features = Φ(state, memoised_features)
                searched_states.append((state, v0, L_BOUND, H(features, θ), features, depth))
            return v0, v1
        alpha = max(alpha, v0)
    if depth > 1:
        # update searched states
        # v0 is EXACT, v1 is EXACT
        features = Φ(state, memoised_features)
        searched_states.append((state, v0, EXACT, H(features, θ), features, depth))
    return v0, v1

def min_value(state, alpha, beta, depth, θ, searched_states, memoised_features=None):
    if state.training_terminal_test():
        return state.utility(train=True), -state.utility(train=True)
    
    v0, v1 = 4*INF, -4*INF
    # we assume whole alpha beta called with even depth, so this is the last min value call
    if depth == 1:
        v1 = H(Φ(state, memoised_features), θ)

    for a in state.actions():
        child = state.result(a)
        pot_v0, pot_v1 = max_value(child, alpha, beta, depth-1, θ, searched_states, memoised_features)
        v1 = max(v1, pot_v1)
        v0 = min(v0, pot_v0)
        if v0 <= alpha:
            if depth > 1:
                # update searched_states
                # v0 is a U_BOUND, v1 is a L_BOUND
                features = Φ(state, memoised_features)
                searched_states.append((state, v1, L_BOUND, H(features, θ), features, depth))
            return v0, v1   
        beta = min(beta, v0)     
    if depth > 1:
        # update searched states
        # v0 is EXACT, v1 is EXACT
        features = Φ(state, memoised_features)
        searched_states.append((state, v1, EXACT, H(features, θ), features, depth))
    return v0, v1

def ab_weight_updates(searched_states, θ, depth, α, λ, MAX_CHANGE):
    Δθ = np.zeros(num_features)
    for state, vs, bound, hs, features, d in searched_states:
        # determine whether we should update
        update = False
        if vs == 0:
            continue
        if (bound == EXACT) or (bound == L_BOUND and vs < hs) or (bound == U_BOUND and vs > hs):
            update = True  
        if not update:
            continue
        
        𝛿 = vs - hs
        Δθ += α*𝛿*features*λ**(depth-d)
        for i in range(num_features):
            if Δθ[i] > MAX_CHANGE:
                Δθ[i] = MAX_CHANGE
            elif Δθ[i] < -MAX_CHANGE:
                Δθ[i] = -MAX_CHANGE
        θ += Δθ

