from alphazero.ConnectFour import ConnectFour
from alphazero.MCTS import MCTS
from alphazero.ResNet import ResNet

from azutil import load_args
import numpy as np
import torch
import sys
import kaggle_environments


class KaggleAgent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args, self.model)

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.col_count)
        state[state == 2] = -1

        state = self.game.change_perspective(state, player)

        policy = self.mcts.search(state)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        if self.args['temperature'] == 0:
            action = int(np.argmax(policy))
        elif self.args['temperature'] == float('inf'):
            action = np.random.choice([r for r in range(self.game.action_size) if policy[r] > 0])
        else:
            policy = policy ** (1 / self.args['temperature'])
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action


def simulate(args, player1_path, player2_path):
    game = ConnectFour()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = ResNet(game, 9, 128, device)
    model1.load_state_dict(torch.load(player1_path, map_location=device), strict=False)
    model1.eval()

    model2 = ResNet(game, 9, 128, device)
    model2.load_state_dict(torch.load(player2_path, map_location=device), strict=False)
    model2.eval()

    env = kaggle_environments.make('connectx')

    player1 = KaggleAgent(model1, game, args)
    player2 = KaggleAgent(model2, game, args)

    players = [player1.run, player2.run]

    env.run(players)

    out = env.render(mode='ansi')

    print(out)


if __name__ == '__main__':
    _, args_path, p1_name, p1_iter, p2_name, p2_iter = sys.argv

    args = load_args(args_path)

    p1_path = f'models/{p1_name}/model_{p1_iter}.pt'
    p2_path = f'models/{p2_name}/model_{p2_iter}.pt'

    simulate(args, p1_path, p2_path)
