import numpy as np
from collections import defaultdict as dd
from copy import deepcopy, copy

BOOM = 'BOOM'
MOVE = 'MOVE'
INF = 30.0
MAX_TURNS = 250
ALL = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
       (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),
       (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),
       (6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7),(7,0),(7,1),(7,2),(7,3),(7,4),(7,5),(7,6),(7,7)]
OPN, DEV, MID, END = 0, 1, 2, 3
HASH = hash

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
            self.board = np.copy(board)
        self.turn = 0
        self.history = dd(int)
        # Stage, num_booms, just_changed_stage?
        self.stage = [OPN, 0, False]

    def actions(self, boom=True):
        curr_piece_adv = self.board[self.board > 0].sum() + self.board[self.board < 0].sum() 
        """Return a list of the allowable moves at this point."""   
        def get_stack_actions(board, x, y, boom):
            ''' Gets the actions of the stack at (x, y) '''
            # all pieces can boom at their position
            actions = [[BOOM, (x, y)]] if boom else []
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
            all_coords = [(x1, y1) for x1, y1 in all_coords if (board[x1][y1] == 0) or (board[x1][y1] > 0)]

            # append all the moves with all possible number of pieces moved
            for n in range(1, height+1):
                for x_dest, y_dest in all_coords:
                    actions.append([MOVE, n, (x, y), (x_dest, y_dest)])

            return actions

        redundant_booms = set()

        actions = []
        for x, y in ALL:
            if self.board[x][y] > 0:
                # if not in redundant_list
                if (x,y) not in redundant_booms and boom:
                    # boom and get board difference, add all exploded white to redundant_list
                    result = self.result(('BOOM', (x,y)))
                    piece_adv = -1*(result.board[result.board > 0].sum() + result.board[result.board < 0].sum())
                    if result.terminal_test() or piece_adv >= curr_piece_adv:
                        actions = [[BOOM, (x, y)]]
                    
                    diff = np.copy(self.board) + result.board
                    for x1, y1 in ALL:
                        if diff[x1][y1] > 0:
                            redundant_booms.add((x1, y1))

                    actions += get_stack_actions(self.board, x, y, boom=True)
                else:
                    actions += get_stack_actions(self.board, x, y, boom=False)

        # Sort such that moves towards opponent's side are first.  Sort by boom then move
        if self.turn%2:
            actions = sorted(actions, key=lambda e: e[3][1] - e[2][1] if e[0] == MOVE else -8 - (8 - e[1][1]))
        else:
            actions = sorted(actions, key=lambda e: e[2][1] - e[3][1] if e[0] == MOVE else -8 -e[1][1])
    
        return actions 

    def result(self, action):
        ''' Return a new state with the action taken from current state '''
        
        new_board = deepcopy(self.board)
        if action[0] == MOVE:
            move, n, orig, dest = action
            x0, y0 = orig
            x1, y1 = dest
            new_board[x0][y0] -= n
            new_board[x1][y1] += n

        if action[0] == BOOM:
            def explode_recursive(board, x0, y0):
                ''' Returns board once explosion has occurred at coordinates '''
                adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
                to_check = set()
                to_check.add((x0,y0))
                ever_seen = set()
                ever_seen.add((x0, y0))
                while len(to_check) > 0:
                    x, y = to_check.pop()
                    neighbours = [(x + dx, y + dy) for dx,dy in adj if 0 <= x+dx <= 7 
                                                                    and 0 <= y+dy <= 7 
                                                                    and (x+dx,y+dy) not in ever_seen
                                                                    and board[x+dx][y+dy]]
                    for coords in neighbours:
                        to_check.add(coords)
                        ever_seen.add(coords)
                    board[x][y] = 0

                return board
            
            x0, y0 = action[1]
            explode_recursive(new_board, x0, y0)
        
        r = State(new_board)
        r.stage = self.stage[:]
        if action[0] == BOOM:
            r.stage[1] += 1
            
        if self.board[self.board > 0].sum() <= 4 and -self.board[self.board < 0].sum() <= 4:
            r.stage[0] = END
        elif r.stage[1] >= 4:
            r.stage[0] = MID
        elif r.stage[1] >= 2:
            r.stage[0] = DEV

        r.stage[2] = r.stage[0] != self.stage[0]
        
        r.board *= -1
        r.turn = self.turn + 1
        r.history = copy(self.history)
        r.history[r] += 1
        return r

    def utility(self):
        """Return the value of this final state."""
        if (self.turn >= MAX_TURNS*2) or (self.history[self] >= 4) or (not (self.board > 0).any() and not (self.board < 0).any()):
            # Draw
            return 0
        if not (self.board < 0).any() and (self.board > 0).any():
            return INF*2
        if not (self.board > 0).any() and (self.board < 0).any():
            return -INF*2 
        piece_adv = self.board[self.board > 0].sum() + self.board[self.board < 0].sum()         
        return np.sign(piece_adv)*INF + piece_adv

        
        

    def terminal_test(self):
        """Return True if this is a final state for the game."""
        # not (black on board and white on board)
        if (self.turn >= MAX_TURNS*2) or (self.history[self] >= 4):
            return True
        return not (self.board < 0).any() or not (self.board > 0).any()

    def stages_terminal_test(self):
        return self.terminal_test() or self.stage[2]

    def training_terminal_test(self):
        num_op = abs(self.board[self.board < 0].sum())
        num_us = self.board[self.board > 0].sum()
        if (num_op == 1 and num_us > 2):
            return True
        if (num_op >= 2 and num_us < 2):
            return True
        if num_op <= 2 and num_us <= 2 and num_op == num_us:
            return True
        return self.terminal_test()

    def display(self):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '\n'.join([''.join([str(space).rjust(3) for space in row]) for row in self.board])

    def __hash__(self): 
        #return HASH(self.board.tostring())
        return self.board.tostring().__hash__()

    def __eq__(self, other):
        return (self.board == other.board).all()