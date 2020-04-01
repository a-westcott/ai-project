import sys
import json

from game import Board, ExpendibotsProblem
from searching import astar_search
from heuristics import h1

GOAL_BOARD = [[0 for _ in range(8)] for _ in range(8)]

def main():
    
    with open(sys.argv[1]) as file:
        data = json.load(file)
    

    from time import time

    start = time()
    '''
        data = {
        "white": [[1,1,4]],
        "black": [[1,4,6]]
    }
    '''
    board_class = Board(data)

    board = board_class.board

    print(board_class)

    problem = ExpendibotsProblem(board, GOAL_BOARD)


    #node = (iterative_deepening_search(problem))
    node = astar_search(problem, h1, True)

    actions = []
    while node.parent:
        actions.append(node.action)
        node = node.parent
    
    for action in actions[::-1]:
        print(board_class.string_action(action))

    end = time()
    print(f'#in {end - start} secs')

if __name__ == '__main__':
    main()
