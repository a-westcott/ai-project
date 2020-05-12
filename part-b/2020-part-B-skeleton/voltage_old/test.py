#%%
def hopefully_better_explode(board, x0, y0):
    adj = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    to_check = set()
    to_check.add((x0,y0))
    ever_seen = set()
    ever_seen.add((x0, y0))
    while len(to_check) > 0:
        x, y = to_check.pop()
        neighbours = [(x + dx, y + dy) for dx,dy in adj if (x+dx,y+dy) not in ever_seen 
                                                        and 0 <= y+dy <= 7 
                                                        and 0 <= x+dx <= 7
                                                        and board[x+dx][y+dy]]
        for coords in neighbours:
            to_check.add(coords)
            ever_seen.add(coords)
        board[x][y] = 0

    return board
#%%
%%timeit 
(hopefully_better_explode([
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0,-1,-1],
                [1, 1, 0, 0, 0, 0,-1,-1]
            ], 0, 0))


# %%
