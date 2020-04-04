SELF = 1
OTHER = -1
BOOM = 0
MOVE = 1


class ExamplePlayer:

    def __init__(self, colour):
        self.colour = colour
        self.board = [
            [1, 1, 0, 0, 0, 0, -1, -1],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, -1, -1],
            [1, 1, 0, 0, 0, 0, -1, -1]
        ]


    def action(self):
        possible_actions = self.get_all_own_actions(self.board)
        from random import choice
        return self.format_action(choice(possible_actions))

    def update(self, colour, action):
        if action[0] == 'MOVE':
            self.move(self.board, action)
        else:
            x0, y0 = action[1]
            self.explode_recursive(self.board, x0, y0)

        # invert the sign of the pieces so that positive has the next move
        self.board = [[-n for n in row] for row in self.board]


    def get_all_own_actions(self, board):
        ''' Returns a list of positive players's possible actions '''
        actions = []
        for x in range(8):
            for y in range(8):
                if board[x][y] > 0:
                    actions += self.get_stack_actions(board, x, y)
        return actions

    def get_stack_actions(self, board, x, y):
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

    def explode(self, board, x, y):
        ''' Returns board once explosion has occurred at coordinates 
        This calls a recursive function once a copy of the board has been made such that the original board is not modified '''
        return self.explode_recursive([row[:] for row in board], x, y, n_explosions=0)
    
    def explode_recursive(self, board, x, y, n_explosions=0):
        ''' Returns board once explosion has occurred at coordinates '''
        # radius is a list of all board positions to blow up
        radius = [(x_, y_) for x_ in range(x-1, x+2) for y_ in range(y-1, y+2) if 0 <= x_ < 8 and 0 <= y_ < 8]
        
        # Try each position
        for x,y in radius:
            # If there's a piece there
            if board[x][y]:
                n_explosions += abs(board[x][y])
                board[x][y] = 0
                board, n_explosions = self.explode_recursive(board, x, y, n_explosions)
            
        return board, n_explosions 

    def move(self, board, action):
        '''
        Move n pieces of a stack from their origin to a destination
        on a board, assuming valid input. Board is mutated in place
        '''
        move, n, orig, dest = action
        x0, y0 = orig
        x1, y1 = dest
        board[x0][y0] -= n
        board[x1][y1] += n

    def format_action(self, action):
        if action[0] == BOOM:
            x,y = action[1]
            statement = ('BOOM', (x,y))
        else:
            n, x, y, x_dest, y_dest = action[1]
            statement = ('MOVE', n, (x,y), (x_dest, y_dest))
        return statement



