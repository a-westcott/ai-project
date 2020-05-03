try:
    from state import State, MOVE, BOOM
    location1 = 'book1.pkl'
    location2 = 'book2.pkl'
except:
    from voltage_mkII.state import State, MOVE, BOOM
    location1 = 'voltage_mkII/book1.pkl'
    location2 = 'voltage_mkII/book2.pkl'


import numpy as np
import pickle

with open(location1, 'rb') as f:
    opening_book = pickle.load(f)

with open(location2, 'rb') as f:
    opening_book2 = pickle.load(f)

opening_book.update(opening_book2)
del opening_book2

# opening_book[state.board.tostring()] = move
