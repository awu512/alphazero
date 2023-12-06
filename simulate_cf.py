import json

from alphazero.ConnectFour import ConnectFour
from alphazero.MCTS import MCTS
from alphazero.ResNet import ResNet

from concurrent.futures import ThreadPoolExecutor, as_completed
from azutil import load_args
from tqdm import tqdm
import numpy as np
import torch
import sys


class Agent:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        self.mcts = MCTS(self.game, self.args, self.model)

    def get_action(self, state, player):
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


class Challenge:
    def __init__(self, game, a1, a2):
        self.game = game
        self.a1 = a1
        self.a2 = a2

    def run(self):
        state = self.game.get_initial_state()
        player = 1

        while True:
            if player == 1:
                action = self.a1.get_action(state, player)
            else:
                action = self.a2.get_action(state, player)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 1:
                    return player
                else:
                    return 0

            player = self.game.get_opponent(player)


def simulate(path, sim_name):
    sim_args = load_args(path)

    game = ConnectFour()
    args = sim_args['args']
    games_per_match = sim_args['games_per_match']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_paths = sim_args['models']
    models = []
    for mpath in model_paths:
        model = ResNet(game, 9, 128, device)
        model.load_state_dict(torch.load(mpath, map_location=device))
        model.eval()
        models.append(model)

    agents = [Agent(model, game, args) for model in models]

    all_results = []

    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents[i+1:]):
            a1_path = model_paths[i]
            a2_path = model_paths[i+j+1]

            results = {
                a1_path: {
                    'won': 0,
                    'lost': 0,
                    'drew': 0
                },
                a2_path: {
                    'won': 0,
                    'lost': 0,
                    'drew': 0
                }
            }

            def run_matchup():
                challenge1 = Challenge(game, a1, a2)
                result1 = challenge1.run()

                challenge2 = Challenge(game, a2, a1)
                result2 = challenge2.run()

                return result1, result2

            game_iters = games_per_match // 2

            with tqdm(total=game_iters) as pbar:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(run_matchup) for _ in range(game_iters)]
                    for future in as_completed(futures):
                        result1, result2 = future.result()

                        if result1 == 1:
                            results[a1_path]['won'] += 1
                        elif result1 == -1:
                            results[a1_path]['lost'] += 1
                        else:
                            results[a1_path]['drew'] += 1

                        if result2 == 1:
                            results[a2_path]['won'] += 1
                        elif result2 == -1:
                            results[a2_path]['lost'] += 1
                        else:
                            results[a2_path]['drew'] += 1

                        pbar.update(1)

            all_results.append({
                'p1': a1_path,
                'p2': a2_path,
                'wins': results[a1_path]['won'],
                'draws': results[a1_path]['drew'],
                'losses': results[a1_path]['lost']
            })

            all_results.append({
                'p1': a2_path,
                'p2': a1_path,
                'wins': results[a2_path]['won'],
                'draws': results[a2_path]['drew'],
                'losses': results[a2_path]['lost']
            })

    with open(f'sim_results/{sim_name}.json', 'w') as f:
        json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    _, args_path, sim_name = sys.argv

    simulate(args_path, sim_name)
