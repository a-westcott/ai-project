try:
    from state import State, MOVE, BOOM
    location1 = 'book1.pkl'
    location2 = 'book2.pkl'
    location3 = 'book3.pkl'
    location4 = 'book4.pkl'
except:
    from voltage.state import State, MOVE, BOOM
    location1 = 'voltage/book1.pkl'
    location2 = 'voltage/book2.pkl'
    location3 = 'voltage/book3.pkl'
    location4 = 'voltage/book4.pkl'


import numpy as np
import pickle

with open(location1, 'rb') as f:
    opening_book = pickle.load(f)

with open(location2, 'rb') as f:
    opening_book2 = pickle.load(f)

with open(location3, 'rb') as f:
    opening_book3 = pickle.load(f)

with open(location4, 'rb') as f:
    opening_book4 = pickle.load(f)

opening_book.update(opening_book2)
opening_book.update(opening_book3)
opening_book.update(opening_book4)

del opening_book2
del opening_book3
del opening_book4

# opening_book[state.board.tostring()] = move
