from alphazero.ConnectFour import ConnectFour
from alphazero.TicTacToe import TicTacToe
from alphazero.ResNet import ResNet
from alphazero.AlphaZeroParallel import AlphaZeroParallel

import json
import torch


def train(game_code):
    hyperparams = get_hyperparameters()

    if game_code == 1:
        game = TicTacToe()
    elif game_code == 2:
        game = ConnectFour()
    else:
        raise Exception(f'Game code [{game_code}] is not valid.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(
        game,
        hyperparams['num_resblocks'],
        hyperparams['num_hidden'],
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    alphazero = AlphaZeroParallel(model, optimizer, game, hyperparams)
    alphazero.learn()


def get_hyperparameters():
    with open('hyperparams.json', 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    game_code = int(input('[1] Tic Tac Toe\n[2] Connect Four\n\nGame to train: '))
    train(game_code)
