import argparse

import torch

from alphazero.AlphaZero import AlphaZero
from alphazero.AlphaZeroParallel import AlphaZeroParallel
from alphazero.ConnectFour import ConnectFour
from alphazero.ResNet import ResNet
from azutil import load_args


def train(args_path, model_name, is_gpu):
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

    if is_gpu:
        alphazero = AlphaZeroParallel(model, optimizer, game, hyperparams)
    else:
        alphazero = AlphaZero(model, optimizer, game, hyperparams)

    alphazero.learn(model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--args', dest='args_path', help='Path to args file')
    parser.add_argument('-n', '--name', dest='model_name', help='Model name')
    parser.add_argument('-gpu', dest='is_gpu', action='store_true', help='Use GPU for self-play')

    cl_args = parser.parse_args()

    train(cl_args.args_path, cl_args.model_name, cl_args.is_gpu)
