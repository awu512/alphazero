import numpy as np


class TicTacToe:
    """ Game definition for TicTacToe """

    def __init__(self):
        self.row_count = 3
        self.col_count = 3
        self.action_size = self.row_count * self.col_count

    def __repr__(self):
        return 'TicTacToe'

    def get_initial_state(self):
        """ Get board with all zeros """
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state, action, player):
        """ Get the next state given the given action by the given player """
        row = action // self.col_count
        col = action % self.col_count
        state[row, col] = player
        return state

    def get_valid_moves(self, state):
        """ Get all the legal moves in the position """
        return (state.reshape(-1) == 0).astype(np.uint8)

    def check_win(self, state, action):
        """ Check if the given action has led to a win """
        if action == None:
            return False

        row = action // self.col_count
        col = action % self.col_count
        player = state[row, col]

        return (
                np.sum(state[row, :]) == player * self.col_count or  # rows
                np.sum(state[:, col]) == player * self.row_count or  # columns
                np.sum(np.diag(state)) == player * self.row_count or  # tl->br diag
                np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count  # tr->bl diag
        )

    def get_value_and_terminated(self, state, action):
        """ Get the value (win/tie) and if the game has terminated """
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        """ Get the opponent of the player """
        return -player

    def get_opponent_value(self, value):
        """ Get the value from the opponent's perspective """
        return -value

    def change_perspective(self, state, player):
        """ Transform the state to be from the opponent's perspective """
        return state * player

    def get_encoded_state(self, state):
        """ Get the network-ready encoded state of the game """
        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(np.float32)

        # check for batched states
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
