import numpy as np
try:
    from state import State
    from player import ALL
except:
    from voltage_mkII.state import State
    from voltage_mkII.player import ALL

RIGHT = 1
LEFT = -1
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
    moves = sorted(state.actions(boom=False), key = lambda move: ((Ox - move[3][0]))**2 + ((Oy - move[3][1])**2)/move[1])

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


class NvTwo():
    '''
    Class for endgame move selection when up n v 2 (n >= 3).
    Using a class as to be able to plan some moves longer term.
    Idea will be to form three stacks along one column 
    and advance towards the opposing stack/s until a boom can 
    be made, in which case we win or we are in n v 1.
    '''
    def __init__(self):
        self.arranged = False
        self.column = None
        self.direction = None
        self.goal_x = [0, 3, 6]
        self.almost = False

    def move(self, state):
        self.state = state
        X_stacks = [(x, y) for x, y in ALL if state.board[x][y] > 0]
        O_stacks = [(x, y) for x, y in ALL if state.board[x][y] < 0]
        
        # form three spaced stacks along starting column if were not there already, checking 
        # were not about to lose a bunch of pieces
        
        # Heuristic
        # Spacing with our pieces
        # Min distance to opponent pieces
        # Opponent's boom in next boom

        # If not spread
            # Maximise spreading ASAP


        # this current arrangement is clunky and im sure it can be improved
        if not self.arranged:
            if self.column is None:
                self.column = 7 if state.turn %2 else 0
                self.direction = LEFT if state.turn %2 else RIGHT
                self.goals = [(x, self.column) for x in self.goal_x]
            
            # fix any immediate threats if we need to

            # if we have < 3 stacks, move current ones to goals if not already there
            #if len(X_stacks) < 3:
            #    for stack in X_stacks:
            #        if stack not in self.goals:
            #            moves = sorted(state.actions(boom=False), key=self.eval_move)
            #            return moves[0]

            # if we have > 3 stacks, move all to its closest goal if not already there
            if len(X_stacks) > 3:
                for stack in X_stacks:
                    if stack not in self.goals:
                        moves = sorted(state.actions(boom=False), key=self.eval_move)
                        return moves[0]
            
            # if we have 3 stacks and goals are filled, self.arranged=True
            self.remain = self.goals.copy()
            for stack in X_stacks:
                if stack in self.remain:
                    self.remain.remove(stack)

            if len(X_stacks) == 3 and len(self.remain) == 0:
                self.arranged = True
                for stack in X_stacks:
                    if stack not in self.goals:
                        self.arranged = False
                        break

            # with the hacky one, everything is on a goal but theyre not filled
            if not self.arranged:
                self.almost = True
                moves = sorted(state.actions(boom=False), key=self.eval_almost_move)
                return moves[0]

        # check if any of our stacks are adj to opponent, boom if so
        for Ox, Oy in O_stacks:
            O_adj = [(Ox + x, Oy + y) for x, y in adj]
            for stack in X_stacks:
                if stack in O_adj:
                    return ('BOOM', stack)

        # otherwise advance
        for x in range(8):
            if state.board[x][self.column] > 0:
                move = ('MOVE', state.board[x][self.column], (x, self.column), (x, self.column + self.direction))
                if x > 5:
                    self.column += self.direction
                return move

    def eval_move(self, move):
        _, n, orig, dest = move
        if orig in self.goals:
            return 10000
        if dest in self.goals:
            return -1*n
        
        x, y = dest
        min_dist = 2000
        for gx, gy in self.goals:
            dist = (x - gx)**2 + (y - gy)**2
            if dist < min_dist:
                min_dist = dist

        return min_dist/n

    def eval_almost_move(self, move):
        _, n, orig, dest = move
        if orig not in self.goals:
            min_dist = 200
            x, y = dest
            for gx, gy in self.remain:
                dist = abs(x - gx)**2 + abs(y - gy)**2
                if dist < min_dist:
                    min_dist = dist

            return -500 + min_dist/n
        x,y = orig
        if orig in self.goals and n != self.state.board[x][y]:
            return 10 + n

        return 200

def main():
    state = State()
    state.board = [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0 ,0],
                [0, 0, 0, 0,-1, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0,-1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]

    thing = NvTwo()
    print(NvTwo().move(state))



if __name__ == '__main__':
    main()