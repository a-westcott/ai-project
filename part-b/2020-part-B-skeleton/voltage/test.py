from zobrist import table, zobrist_index, intial_board_hash

import numpy as np

#%%
%timeit hash('123')


# %%
board = np.array([
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1]
            ])
%timeit hash(board.tostring())

# %%
import numpy as np
from zobrist import table, zobrist_index, intial_board_hash
h = intial_board_hash()

def move(h):
    h = np.bitwise_xor(h, table[0][0][zobrist_index(1)])
    h = np.bitwise_xor(h, table[0][1][zobrist_index(1)])
    h = np.bitwise_xor(h, table[0][1][zobrist_index(2)])



# %%
%timeit move(h)


# %%
%timeit np.bitwise_xor(10546568963754086732, 16543513965714919198)

# %%
