SELF = 1
OTHER = -1
BOOM = 0
MOVE = 1
INF = 9999.0

import numpy as np
from copy import deepcopy

class State():
    def __init__(self, board=None):
        if board is None:
            self.board = np.array([
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1]
            ])
        else:
            self.board = deepcopy(board)

    def actions(self):
        """Return a list of the allowable moves at this point."""   
        def get_stack_actions(board, x, y):
            ''' Gets the actions of the stack at (x, y) '''
            # all pieces can boom at their position
            colour = OTHER if board[x][y] < 0 else SELF
            actions = [[BOOM, (x, y)]]
            # Moves
            height = abs(board[x][y])

            # all possible move positions on the board, including moves on to oposing pieces
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
            all_coords = [(x1, y1) for x1, y1 in all_coords if (board[x1][y1] == 0) or (board[x1][y1]*colour > 0)]

            # append all the moves with all possible number of pieces moved
            for n in range(1, height+1):
                for x_dest, y_dest in all_coords:
                    actions.append([MOVE, (n, x, y, x_dest, y_dest)])

            return actions

        actions = []
        for x in range(8):
            for y in range(8):
                if self.board[x][y] > 0:
                    actions += get_stack_actions(self.board, x, y)
        return actions 

    def result(self, action):
        ''' Return a new state with the action taken from current state 
            MOST DEFINNITELY NEEDS TO BE FIXED
        
        '''
        
        new_board = deepcopy(self.board)
        if action[0] == 'MOVE':
            move, n, orig, dest = action
            x0, y0 = orig
            x1, y1 = dest
            new_board[x0][y0] -= n
            new_board[x1][y1] += n

        if action[0] == MOVE:
            n, x0, y0, x1, y1 = action[1]
            new_board[x0][y0] -= n
            new_board[x1][y1] += n
        
        if action[0] == 'BOOM':
            def explode_recursive(board, x, y, n_explosions=0):
                ''' Returns board once explosion has occurred at coordinates '''
                # radius is a list of all board positions to blow up
                radius = [(x_, y_) for x_ in range(x-1, x+2) for y_ in range(y-1, y+2) if 0 <= x_ < 8 and 0 <= y_ < 8]
                
                # Try each position
                for x,y in radius:
                    # If there's a piece there
                    if board[x][y]:
                        n_explosions += abs(board[x][y])
                        board[x][y] = 0
                        board, n_explosions = explode_recursive(board, x, y, n_explosions)
            
                return board, n_explosions 
            
            x0, y0 = action[1]
            explode_recursive(new_board, x0, y0)

        if action[0] == BOOM:
            def explode_recursive(board, x, y, n_explosions=0):
                ''' Returns board once explosion has occurred at coordinates '''
                # radius is a list of all board positions to blow up
                radius = [(x_, y_) for x_ in range(x-1, x+2) for y_ in range(y-1, y+2) if 0 <= x_ < 8 and 0 <= y_ < 8]
                
                # Try each position
                for x,y in radius:
                    # If there's a piece there
                    if board[x][y]:
                        n_explosions += abs(board[x][y])
                        board[x][y] = 0
                        board, n_explosions = explode_recursive(board, x, y, n_explosions)
            
                return board, n_explosions 
            
            x0, y0 = action[1]
            explode_recursive(new_board, x0, y0)
            
        return State(new_board)


    def utility(self):
        """Return the value of this final state."""
        if self.terminal_test():
            # Draw
            if not (self.board < 0).any() and not (self.board < 0).any():
                return 0
            if (self.board > 0).any():
                return INF
            return -INF
        return self.board[self.board > 0].sum() / -self.board[self.board < 0].sum()

    def terminal_test(self):
        """Return True if this is a final state for the game."""
        # not (black on board and white on board)
        return not (self.board < 0).any() or not (self.board > 0).any()

    def display(self):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '#' + '\n#'.join([''.join([str(space).rjust(3) for space in row]) for row in self.board])


    '''
    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))
    '''

    
class BasePlayer:
    def __init__(self, colour):
        self.colour = colour
        self.state = State()
        
    def action(self):
        possible_actions = self.state.actions()
        from random import choice
        return self.format_action(choice(possible_actions))

    def update(self, colour, action):    
        self.state = self.state.result(action) 

        
        # invert the sign of the pieces so that positive has the next move
        self.state.board = -1*self.state.board

    def format_action(self, action):
        if action[0] == BOOM:
            x,y = action[1]
            statement = ('BOOM', (x,y))
        else:
            n, x, y, x_dest, y_dest = action[1]
            statement = ('MOVE', n, (x,y), (x_dest, y_dest))
        return statement


    
class StandardPlayer(BasePlayer):
    def action(self):
        return self.format_action(self.negamax_search(self.state))

    def negamax_search(self, state, depth=1, cutoff_test=None, eval_fn=None):
        """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

        # Functions used by alpha_beta
        def max_value(state, alpha, beta, depth):
            if state.terminal_test() or depth == 0:
                return state.utility()
            v = -INF
            for a in state.actions():
                child = state.result(a)
                # game state must be flipped
                v = max(v, max_value(State(-1*child.board), best_score, beta, depth-1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        # Body of alpha_beta_cutoff_search starts here:
        # The default test cuts off at depth d or at a terminal state
        best_score = -INF
        beta = +INF
        best_action = None
        for a in state.actions():
            child = state.result(a)
            print(state)
            print()
            print(child)
            # game state must be flipped
            v = max_value(State(-1*child.board), best_score, beta, depth-1)
            print(a, max_value(State(-1*child.board), best_score, beta, depth-1), child.utility())
            if v > best_score:
                best_score = v
                best_action = a
        return best_action