import unittest

from game import Board, BOOM

class TestCases(unittest.TestCase):
    
    def test(self):
        test1 = Board()
        # The 11 white stack can't move, extra 1 white at (0, 0)
        test2 = Board({'white': [[11, 4, 4], [1, 0, 0]], 
                       'black': [[1, 0, 4], [1, 1, 4], [1, 2, 4], [1, 3, 4], [1, 5, 4], [1, 6, 4], [1, 7, 4],
                                 [1, 4, 0], [1, 4, 1], [1, 4, 2], [1, 4, 3], [1, 4, 5], [1, 4, 6], [1, 4, 7]]})
        # White stack in corner that can move
        test3 = Board({'white': [[4, 7, 7]],
                       'black': [[3, 7, 6], [3, 6, 7], [2, 7, 5], [2, 5, 7], [1, 7, 4], [1, 4, 7]]})
        # Board full of white pieces
        test4 = Board({'white': [[1, x, y] for x in range(8) for y in range(8)], 'black': []})
        
        # __init__(self, stacks) & place_stacks(self, stacks) 
        self.assertEqual(test1.board, [[0]*8]*8)
        self.assertEqual(test2.board, [[1,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], 
                                       [-1,-1,-1,-1,11,-1,-1,-1], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0]])
        self.assertEqual(test2.n_white, 12)
        self.assertEqual(test2.n_black, 14)
        
        # get_all_white_actions(self, board)
        self.assertEqual(test1.get_all_white_actions(test1.board), [])
        self.assertEqual(test2.get_all_white_actions(test2.board), 
                         [[BOOM, (0, 0)], [1, (1, 0, 0, 1, 0)], [1, (1, 0, 0, 0, 1)], [BOOM, (4, 4)]])
        self.assertEqual(test3.get_all_white_actions(test3.board), 
                         [[0, (7, 7)], [1, (1,7,7,3,7)], [1, (1,7,7,7,3)], [1, (2,7,7,3,7)], [1, (2,7,7,7,3)], 
                          [1, (3,7,7,3,7)], [1, (3,7,7,7,3)], [1, (4,7,7,3,7)], [1, (4,7,7,7,3)]])
        
        # get_actions(self, board, x, y)
        self.assertEqual(test3.get_actions(test3.board, 7, 5), 
                         [[0, (7, 5)], [1, (1,7,5,6,5)], [1, (1,7,5,5,5)], [1, (1,7,5,7,6)], [1, (1,7,5,7,4)], [1, (1,7,5,7,3)],
                          [1, (2,7,5,6,5)], [1, (2,7,5,5,5)], [1, (2,7,5,7,6)], [1, (2,7,5,7,4)], [1, (2,7,5,7,3)]])

        # explode(self, board, x, y)
        self.assertEqual(test4.explode(test4.board, 0, 0), ([[0]*8]*8, 64))
        self.assertEqual(test2.explode(test2.board, 7, 4), ([[1] + [0]*7] + [[0]*8]*7, 25))
        
        # move(self, board, n, x1, y1, x2, y2)
        self.assertEqual(test3.move(test3.board, 1, 6, 7, 6, 5), 
                         [[0]*8]*4 + [[0]*7+[-1], [0]*7+[-2], [0]*5+[-1,0,-2], [0]*4+[-1,-2,-3,4]])
        self.assertEqual(test3.move(test3.board, 3, 6, 7, 4, 7), 
                         [[0]*8]*4 + [[0]*7+[-4], [0]*7+[-2], [0]*8, [0]*4+[-1,-2,-3,4]])
        
        # __str__(self)
        
if __name__ == '__main__':
    unittest.main()