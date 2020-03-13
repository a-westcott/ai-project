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

    data = {"white": [[1,0,0]], "black": [[1,0,1]]}


    board_class = Board(data)

    board = board_class.board

    print(board_class)

    problem = ExpendibotsProblem(board, GOAL_BOARD)
    print(iterative_deepening_search(problem))


if __name__ == '__main__':
    main()
