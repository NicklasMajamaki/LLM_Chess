class ParseException(Exception):
    """ Exception raised when model generates an output that cannot be parsed. """
    pass

class IllegalMoveException(Exception):
    """ Exception raised when a move is invalid. """
    pass