import numpy as np


class ConnectFour:
    """ Game definition for ConnectFour """

    def __init__(self):
        self.row_count = 6
        self.col_count = 7
        self.action_size = self.col_count
        self.win_condition = 4

    def __repr__(self):
        return 'ConnectFour'

    def get_initial_state(self):
        """ Get board with all zeros """
        return np.zeros((self.row_count, self.col_count))

    def get_next_state(self, state, action, player):
        """ Get the next state given the given action by the given player """
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        """ Get all the legal moves in the position """
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        """ Check if the given action has led to a win """
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.win_condition):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                        r < 0
                        or r >= self.row_count
                        or c < 0
                        or c >= self.col_count
                        or state[r][c] != player
                ):
                    return i - 1
            return self.win_condition - 1

        return (
                count(1, 0) >= self.win_condition - 1  # vertical
                or (count(0, 1) + count(0, -1)) >= self.win_condition - 1  # horizontal
                or (count(1, 1) + count(-1, -1)) >= self.win_condition - 1  # top left diagonal
                or (count(1, -1) + count(-1, 1)) >= self.win_condition - 1  # top right diagonal
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
