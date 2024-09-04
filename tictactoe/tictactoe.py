"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_counter = 0
    o_counter = 0

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == X:
                x_counter += 1
            elif board[i][j] == O:
                o_counter += 1

    if x_counter > o_counter:
        return O
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    result = copy.deepcopy(board)
    result[action[0]][action[1]] = player(board)
    return result


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    n = len(board)
    for i in range(n):
        if all(board[i][j] == "X" for j in range(n)):
            return "X"
        if all(board[i][j] == "O" for j in range(n)):
            return "O"
        if all(board[j][i] == "X" for j in range(n)):
            return "X"
        if all(board[j][i] == "O" for j in range(n)):
            return "O"

    if all(board[i][i] == "X" for i in range(n)):
        return "X"
    if all(board[i][i] == "O" for i in range(n)):
        return "O"
    if all(board[i][n - i - 1] == "X" for i in range(n)):
        return "X"
    if all(board[i][n - i - 1] == "O" for i in range(n)):
        return "O"

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    if any(EMPTY in row for row in board):
        return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == "X":
            return 1
        elif winner(board) == "O":
            return -1
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    else:
        if player(board) == X:
            value, move = max_value(board)
            return move
        else:
            value, move = min_value(board)
            return move


def max_value(board):
    if terminal(board):
        return utility(board), None

    v = float("-inf")
    move = None
    for action in actions(board):
        aux, act = min_value(result(board, action))
        if aux > v:
            v = aux
            move = action
            if v == 1:
                return v, move

    return v, move


def min_value(board):
    if terminal(board):
        return utility(board), None

    v = float("inf")
    move = None
    for action in actions(board):
        aux, act = max_value(result(board, action))
        if aux < v:
            v = aux
            move = action
            if v == -1:
                return v, move

    return v, move
