'''
A bunch of different heuristics we can try for A* search
'''

# max(Manhattan distance from all black to nearest white)

def h0(node):
    ''' Takes in board, returns an approximation for cost to completion '''

    board = node.state

    black_pieces = [(x, y) for x in range(8) for y in range(8) if board[x][y] < 0]
    white_pieces = [(x, y) for x in range(8) for y in range(8) if board[x][y] > 0]
    
    farthest = 0
    for x1, y1 in black_pieces:
        closest = 100
        for x2, y2 in white_pieces:
            closest = min(closest, abs(x1-x2) + abs(y1-y2))

        farthest = max(farthest, closest)
    
    return farthest

def h1(node):
    ''' Takes in board, returns an approximation for cost to completion '''

    board = node.state
    
    black_pieces = [(x, y) for x in range(8) for y in range(8) if board[x][y] < 0]
    white_pieces = [(x, y) for x in range(8) for y in range(8) if board[x][y] > 0]
    
    total = 0
    for x1, y1 in black_pieces:
        closest = 100
        for x2, y2 in white_pieces:
            closest = min(closest, abs(x1-x2) + abs(y1-y2))

        total += closest
    
    return total


def h2(node):
    '''
    combine h0 and h1, weighted evenly
    '''
    return h0(node) + h1(node)

