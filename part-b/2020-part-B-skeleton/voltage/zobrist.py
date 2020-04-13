import numpy as np

MAX_UINT64 = np.iinfo(np.uint64).max

DIM = 8
NO_PIECES = 24

WHITE, BLACK = 1, -1
INITIAL_BOARD = np.array([
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1]
            ])

def rand_uint64():
    return np.random.randint(0, MAX_UINT64, None, np.uint64)

def generate_table():
    table = np.empty((DIM, DIM, NO_PIECES), np.uint64)
    for x in range(DIM):
        for y in range(DIM):
            for piece in range(NO_PIECES):
                table[x][y][piece] = rand_uint64()
    return table


def zobrist_index(piece):
    '''
    returns the index in the zobrist table for the given piece
    '''
    if piece > 0:
        # positive pieces are 0, 2, 4,...,22
        return 2*(piece - 1)
    if piece < 0:
        # negative are 1, 3, 5,...,23
        return 2*piece + 1

try:
    table = np.load('zobrist_table.npy')
except:
    if __name__ != '__main__':
        error = "Table not generated, need to run zobrist.py"
        print()
        print('!'*len(error))
        print(error)
        print('!'*len(error))
        print()

def intial_board_hash(player=WHITE, table=table):
    '''
    Given a defined zobrist table, return the hash of the initial board
    '''
    h = np.uint64(0)
    board = INITIAL_BOARD*player
    for x in range(DIM):
        for y in range(DIM):
            if board[x][y]:
                index = zobrist_index(board[x][y])
                h = np.bitwise_xor(h, table[x][y][index])
    return h


def main():
    '''
    Initilisation of a zobrist hashing scheme for expendibots
    '''

    print(intial_board_hash())

    #table = generate_table()
    #np.save('zobrist_table', table)
    

if __name__ == '__main__':
    main()