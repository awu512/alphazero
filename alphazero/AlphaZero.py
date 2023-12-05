from alphazero.MCTS import MCTS

import numpy as np
import random
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F

class AlphaZero:
    """ AlphaZero class for self-play and training """

    def __init__(self, model, optimizer, game, args):
        """ Initialize the AlphaZero instance """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def self_play(self):
        """ Run a single self-play game until completion and generate outcome-appended training data """

        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            # get the current state and action probabilities from MCTS
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            # record a game snapshot
            memory.append((neutral_state, action_probs, player))

            # randomly sample an action from the distribution
            adjusted_action_probs = action_probs ** (
                        1 / self.args['temperature'])  # add flexibility for exploration / exploitation
            adjusted_action_probs /= np.sum(adjusted_action_probs)
            action = np.random.choice(self.game.action_size, p=adjusted_action_probs)

            # get the next state given the chosen action
            state = self.game.get_next_state(state, action, player)

            # check for game completion
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # get all states and policies from the game and append the outcome
                return [(
                    self.game.get_encoded_state(h_state),
                    h_action_probs,
                    value if h_player == player else self.game.get_opponent_value(value)
                ) for h_state, h_action_probs, h_player in memory]

            # swap the player and loop
            player = self.game.get_opponent(player)

    def train(self, memory):
        """ Train the model """

        # randomize training data
        random.shuffle(memory)

        for batch_i in range(0, len(memory), self.args['batch_size']):
            # sample a batch from training data
            sample = memory[batch_i: min(len(memory) - 1, batch_i + self.args['batch_size'])]

            # transpose list of tuples to independent lists
            try:
                state, policy_targets, value_targets = zip(*sample)
            except ValueError as e:
                print(e)
                continue

            # convert to numpy arrays
            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)  # wrap each value in its own array for simplicity

            # convert to tensors
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            # get model outputs
            out_policy, out_value = self.model(state)

            # get loss
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # minimize loss via backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self, model_name):
        """ Generate self-play training data and train the model on it """

        Path(f'models/{model_name}').mkdir(parents=True, exist_ok=True)

        for iter in range(self.args['num_iters']):
            memory = []

            self.model.eval()
            with tqdm(total=self.args['num_self_play_iters']) as pbar:
                with ThreadPoolExecutor(max_workers=self.args['num_threads']) as executor:
                    futures = [executor.submit(self.self_play) for _ in range(self.args['num_self_play_iters'])]
                    for future in as_completed(futures):
                        memory += future.result()
                        pbar.update(1)

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f'models/{model_name}/model_{iter}.pt')
            torch.save(self.optimizer.state_dict(), f'models/{model_name}/optimizer_{iter}.pt')