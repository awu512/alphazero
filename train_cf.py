import sys

import torch

from alphazero.AlphaZeroParallel import AlphaZeroParallel
from alphazero.ConnectFour import ConnectFour
from alphazero.ResNet import ResNet
from azutil import load_args


def train(args_path, model_name):
    hyperparams = load_args(args_path)

    game = ConnectFour()

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
    alphazero.learn(model_name)


if __name__ == '__main__':
    _, args_path, model_name = sys.argv

    train(args_path, model_name)
