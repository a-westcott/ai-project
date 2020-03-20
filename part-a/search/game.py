from searching import Problem

BOOM = 0
MOVE = 1
BLACK = -1
WHITE = 1

class Board:
    ''' An Expendibots board '''
    
    def __init__(self, stacks=None):
        self.board = [[0 for _ in range(8)] for _ in range(8)]
        
        self.n_white = 0
        self.n_black = 0

        if stacks is not None:
            self.place_stacks(stacks)        
    
    def place_stacks(self, stacks):
        ''' 
        Initialises board with stacks from dictionary "stacks" 
        White stacks are positive numbers
        Black stacks are negative numbers
        '''
        white_stacks = stacks['white']
        black_stacks = stacks['black']

        for stack in white_stacks:
            n,x,y = stack
            self.board[x][y] = n
            self.n_white += n
            
        for stack in black_stacks:
            n,x,y = stack
            self.board[x][y] = -n
            self.n_black += n

    def get_all_white_actions(self, board):
        ''' Returns a list of white's possible actions '''
        actions = []
        for x in range(8):
            for y in range(8):
                if board[x][y] > 0:
                    actions += self.get_actions(board, x, y)
        return actions

    def get_actions(self, board, x, y):
        ''' Gets the actions of the stack at (x, y) '''
        # all pieces can boom at their position
        colour = BLACK if board[x][y] < 0 else WHITE
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
    

    
    def get_explode_positions(self):
        ''' Returns a list of coordinates for the AI to explode at 
        #################################################################################################
        #################### THIS FUNCTION IS DEPRECATED, KILL IT, BURN IT WITH FIRE ####################
        #################################################################################################
        '''
        # only_black_board is the board with only black pieces
        only_black_board = [[(lambda stack: stack if stack <= 0 else 0)(stack) for stack in row] for row in self.board]
        # Unexploded is a list of black pieces that are not yet in range by current planned explosions
        unexploded = [(x, y) for x in range(8) for y in range(8) if only_black_board[x][y] < 0]
        # Planned is a list of coordinates where we plan to explode
        planned = []

        while unexploded:
            biggest = 0
            # Find coordinates that maximise explosions
            for x in range(8):
                for y in range(8):
                    if only_black_board[x][y] == 0:
                        tmp = [row[:] for row in only_black_board]
                        tmp, n_explosions = self.explode(tmp, x, y)


    def explode(self, board, x, y):
        ''' Returns board once explosion has occurred at coordinates 
        This calls a recursive function once a copy of the board has been made such that the original board is not modified '''
        return self.explode_recursive([row[:] for row in board], x, y, n_explosions=0)
    
    def explode_recursive(self, board, x, y, n_explosions):
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
            

    def move(self, board, n, x1, y1, x2, y2):
        '''
        Moves a specified number of pieces of a stack to a specified spot
        Assumes move is valid
        '''
        board_copy = [row[:] for row in board]
        
        if board_copy[x1][y1] > 0:
            board_copy[x1][y1] -= n
            board_copy[x2][y2] += n
        # black stacks have negative values
        elif board_copy[x1][y1] < 0:
            board_copy[x1][y1] += n
            board_copy[x2][y2] -= n

        return board_copy

    def __str__(self):
        # sorry cameron i runied it and i dont understand the magic to make it nice
        return '#' + '\n#'.join([''.join([str(space).rjust(3) for space in row]) for row in self.board])

    def string_action(self, action):
        statement = ''
        if action[0] == BOOM:
            x,y = action[1]
            statement = f'BOOM at ({x}, {y}).'
        else:
            n, x, y, x_dest, y_dest = action[1]
            statement = f'MOVE {n} from ({x}, {y}) to ({x_dest}, {y_dest}).'
        return statement


class ExpendibotsProblem(Problem):
    def actions(self, state):
        return Board().get_all_white_actions(state)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        
        action_type = action[0]
        if action_type == MOVE:
            n, x1, y1, x2, y2 = action[1]
            return Board().move(state, n, x1, y1, x2, y2)
        x, y = action[1]
        board, n_explosions = Board().explode(state, x, y)
        return board

