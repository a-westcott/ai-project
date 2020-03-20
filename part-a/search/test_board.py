import unittest

from game import Board, BOOM, MOVE

class TestCases(unittest.TestCase):
    
    def setUp(self):
        self.test_board1 = Board()
        # The 11 white stack can't move, extra 1 white at (0, 0)
        self.test_board2 = Board({'white': [[11, 4, 4], [1, 0, 0]], 
                       'black': [[1, 0, 4], [1, 1, 4], [1, 2, 4], [1, 3, 4], [1, 5, 4], [1, 6, 4], [1, 7, 4],
                                 [1, 4, 0], [1, 4, 1], [1, 4, 2], [1, 4, 3], [1, 4, 5], [1, 4, 6], [1, 4, 7]]})
        # White stack in corner that can move
        self.test_board3 = Board({'white': [[4, 7, 7]],
                       'black': [[3, 7, 6], [3, 6, 7], [2, 7, 5], [2, 5, 7], [1, 7, 4], [1, 4, 7]]})
        # Board full of white pieces
        self.test_board4 = Board({'white': [[1, x, y] for x in range(8) for y in range(8)], 'black': []})


    def test_board_initilisation(self):
        self.assertEqual(self.test_board1.board, [[0]*8]*8)
        self.assertEqual(self.test_board2.board, [[1,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], 
                                       [-1,-1,-1,-1,11,-1,-1,-1], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,-1,0,0,0]])
        
        self.assertEqual(self.test_board2.n_white, 12)
        self.assertEqual(self.test_board2.n_black, 14)


    def test_get_all_white_actions(self):
        self.assertEqual(self.test_board1.get_all_white_actions(self.test_board1.board), [])
        self.assertEqual(self.test_board2.get_all_white_actions(self.test_board2.board), 
                         [[BOOM, (0, 0)], [1, (1, 0, 0, 1, 0)], [1, (1, 0, 0, 0, 1)], [BOOM, (4, 4)]])
        self.assertEqual(self.test_board3.get_all_white_actions(self.test_board3.board), 
                         [[0, (7, 7)], [1, (1,7,7,3,7)], [1, (1,7,7,7,3)], [1, (2,7,7,3,7)], [1, (2,7,7,7,3)], 
                          [1, (3,7,7,3,7)], [1, (3,7,7,7,3)], [1, (4,7,7,3,7)], [1, (4,7,7,7,3)]])
        
    def test_get_actions(self):
        self.assertEqual(self.test_board3.get_actions(self.test_board3.board, 7, 5), 
                         [[0, (7, 5)], [1, (1,7,5,6,5)], [1, (1,7,5,5,5)], [1, (1,7,5,7,6)], [1, (1,7,5,7,4)], [1, (1,7,5,7,3)],
                          [1, (2,7,5,6,5)], [1, (2,7,5,5,5)], [1, (2,7,5,7,6)], [1, (2,7,5,7,4)], [1, (2,7,5,7,3)]])


    def test_explode(self):
        self.assertEqual(self.test_board4.explode(self.test_board4.board, 0, 0), ([[0]*8]*8, 64))
        self.assertEqual(self.test_board2.explode(self.test_board2.board, 7, 4), ([[1] + [0]*7] + [[0]*8]*7, 25))
        
    def test_move(self):
        self.assertEqual(self.test_board3.move(self.test_board3.board, 1, 6, 7, 6, 5), 
                         [[0]*8]*4 + [[0]*7+[-1], [0]*7+[-2], [0]*5+[-1,0,-2], [0]*4+[-1,-2,-3,4]])
        self.assertEqual(self.test_board3.move(self.test_board3.board, 3, 6, 7, 4, 7), 
                         [[0]*8]*4 + [[0]*7+[-4], [0]*7+[-2], [0]*8, [0]*4+[-1,-2,-3,4]])
        
    def test_print_action_boom(self):
        string_action = Board().string_action
        self.assertEqual(string_action([BOOM, (0, 0)]), "BOOM at (0, 0).")
        self.assertEqual(string_action([BOOM, (5, 2)]), "BOOM at (5, 2).")


    def test_print_action_move(self):
        string_action = Board().string_action
        self.assertEqual(string_action([MOVE, (1, 0, 0, 0, 1)]), "MOVE 1 from (0, 0) to (0, 1).")
        self.assertEqual(string_action([MOVE, (5, 2, 3, 7, 3)]), "MOVE 5 from (2, 3) to (7, 3).")



        
if __name__ == '__main__':
    unittest.main()