from MCTS import Node

import numpy as np
import torch


class MCTSParallel:
    """ A parallelized Monte Carlo Tree Search """

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, sp_games):
        """ Expand and explore the MCTS and update value sums and visit counts """

        # add some random noise to policy to increase exploration
        policy, _ = self.model(torch.tensor(self.game.get_encoded_state(states), device=self.model.device))
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) *
                  policy +
                  self.args['dirichlet_epsilon'] *
                  np.random.dirichlet([self.args['dirichlet_alpha']] *
                                      self.game.action_size,
                                      size=policy.shape[0]
                                      ))

        for i, spg in enumerate(sp_games):
            # get policy for this self-play game
            spg_policy = policy[i]

            # mask illegal moves out of policy
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)

            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in sp_games:
                spg.node = None
                node = spg.root

                # SELECTION
                while node.is_expanded():
                    node = node.select()

                # check for end of game
                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

                # flip parent value
                value = self.game.get_opponent_value(value)

                # backpropagate value sums and visit counts if game ended
                if is_terminal:
                    node.backpropagate(value)

                # otherwise, store the current node
                else:
                    spg.node = node

            expandable_sp_games = [map_i for map_i in range(len(sp_games)) if sp_games[map_i].node is not None]

            if len(expandable_sp_games) > 0:
                states = np.stack([sp_games[map_i].node.state for map_i in expandable_sp_games])

                # get output from model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )

                # change policy to proabability distribution
                policy = torch.softmax(policy, axis=1).cpu().numpy()

                # get value as numpy
                value = value.cpu().numpy()

            for i, map_i in enumerate(expandable_sp_games):
                # get the current node for the self-play game
                node = sp_games[map_i].node

                # get policy and value from self-play game
                spg_policy, spg_value = policy[i], value[i]

                # mask out illegal moves
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves

                # readjust back to probability distribution
                spg_policy /= np.sum(spg_policy)

                # expand the node
                node.expand(spg_policy)

                # backpropagate values and visit counts
                node.backpropagate(spg_value)