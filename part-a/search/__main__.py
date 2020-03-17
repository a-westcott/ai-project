import sys
import json

from util import print_move, print_boom, print_board
from game import Board, ExpendibotsProblem
from searching import iterative_deepening_search

GOAL_BOARD = [[0 for _ in range(8)] for _ in range(8)]

def main():
    '''
    with open(sys.argv[1]) as file:
        data = json.load(file)
    ''' 

    from time import time

    start = time()

    data = {
    "white": [[1,3,5],[1,4,3]],
    "black": [[1,0,7],[1,4,1],[1,6,2],[1,7,3]]}

    board_class = Board(data)

    board = board_class.board

    print(board_class)

    problem = ExpendibotsProblem(board, GOAL_BOARD)

    node = (iterative_deepening_search(problem))

    actions = []
    while node.parent:
        actions.append(node.action)
        node = node.parent
    print(actions[::-1])


    end = time()

    print(f'in {end - start} secs')

if __name__ == '__main__':
    main()
