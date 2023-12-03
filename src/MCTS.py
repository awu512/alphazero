import math
import numpy as np
import torch

class MCTS:
    """ A Monte Carlo Tree Search """

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        """ Expand and explore the MCTS and update value sums and visit counts """
        # DEFINE ROOT
        root = Node(self.game, self.args, state, visit_count=1)

        # add some random noise to policy to increase exploration
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) *
                  policy +
                  self.args['dirichlet_epsilon'] *
                  np.random.dirichlet([self.args['dirichlet_alpha']] *
                                      self.game.action_size
                                      ))
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            # SELECTION
            while node.is_expanded():
                node = node.select()

            # check for end of game
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

            # flip parent value
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # get output from model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )

                # change policy to proabability distribution
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

                # mask out illegal moves
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves

                # readjust back to probability distribution
                policy /= np.sum(policy)

                # get the value as a number from singleton tensor
                value = value.item()

                # EXPANSION
                node.expand(policy)

            # BACKPROP
            node.backpropagate(value)

        # probabilities of action being good
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs

class Node:
    """ A Monte Carlo Tree Search node """

    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_expanded(self):
        """ Check if the node already has children. Since we expand all children of a node at once, we check for >0 children """
        return len(self.children) > 0

    def select(self):
        """ Select a child to explore """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        """ Get how promising a move is from the opponent's perspective, normalized on [0,1] """
        if child.visit_count == 0:
            q = 0
        else:
            q = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        return q + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self, policy):
        """ Expand a node by adding all legal child moves """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        """ Propagate value sums and visit counts from children to all parents """
        self.value_sum += value
        self.visit_count += 1

        # flip value for opponent (parent)
        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropagate(value)