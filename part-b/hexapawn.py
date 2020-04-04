INF = 1
X, O, BLANK = 'X', 'O', ' '
N = 6
DOWN, UP = 1, -1

class State():
    
    UTILITY = {X: INF, O: -INF, None: 0}
    
    def __init__(self, turn, state=None, N=N):
        self.board = X*N + BLANK*(N-2)*N + O*N
        self.N = N
        
        if state is not None:
            if type(state) is str:
                self.board = state
            else:
                self.board = state.board
        
        self.turn = turn
            
    def utility(self):
        ''' Returns the utility of a particular state '''
        # If terminal state, score accordingly
        if self.terminal_test() is not None:
            if self.terminal_test() == X:
                return INF
            return -INF
        
        # Else return score according to piece count ratio
        return (self.board.count(X)/(self.board.count(O)+1))/10   
            
    def move(self, move):
        ''' Moves players piece from Start to End index '''
        s, e = move
        player = self.board[s]
        # Remove piece
        self.board = self.board[:s] + BLANK + self.board[s+1:]
        # Place piece
        self.board = self.board[:e] + player + self.board[e+1:] 
        self.turn = X if self.turn == O else O
    
    def terminal_test(self):
        ''' Returns the victorious player if there is one '''
        # All pieces taken
        if X not in self.board:
            return O
        if O not in self.board:
            return X
        
        # Check if either player has crossed board
        if X in self.board[self.N*(self.N-1):]:
            return X
        if O in self.board[:self.N]:
            return O
        
        # Check if player cannot move, (return opponent as winner if so)
        if len(self.actions()) == 0:
            return X if self.turn == O else O
        
        return None
            
    def __str__(self):        
        return (2*self.N+1)*'_' + ''.join(['\n|' + ''.join([e + '|' for e in self.board[i:i+self.N]]) + ' ' + \
                                     ''.join([str(j).rjust(3) for j in range(i, i+self.N)]) for i in range(0, self.N**2, self.N)])    
    
    def __repr__(self):
        return str(self)
    
    def actions(self):
        ''' Returns a list of possible actions for the player '''
        player   = X if self.turn == X else O
        opponent = O if self.turn == X else X
        direction = DOWN if player == X else UP
    
        # Find pieces
        lst = [i for i in range(self.N**2) if self.board[i] == player]

        actions = []
        for i in lst:
            # Check vertical is valid and free
            if (0 <= i+self.N*direction < self.N**2) and (self.board[i+self.N*direction] == BLANK):
                actions.append((i, i+self.N*direction))

            # Check diagonals are valid and taken by opposition
            # If not on left, check left
            new_i = i + self.N*direction - 1
            if (i%self.N != 0) and (0 <= new_i) and (self.board[new_i] == opponent):
                actions.append((i, new_i))

            # If not on right, check right
            new_i = i + self.N*direction + 1
            if ((i+1)%self.N != 0) and (new_i < self.N**2) and (self.board[new_i] == opponent):
                actions.append((i, new_i))

        return actions