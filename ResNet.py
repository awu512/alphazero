import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    """ The residual neural network implementation for AlphaZero """

    def __init__(self, game, num_resblocks, num_hidden, device):
        super().__init__()

        # store the specified device for computation
        self.device = device

        # initial NN block
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        # shared portion of network between policy and value heads
        self.backbone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resblocks)]
        )

        # the portion of the network responsible for outputting policies
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.col_count, game.action_size)
        )

        # the portion of the network responsible for outputting values
        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.col_count, 1),
            nn.Tanh()
        )

        # send computation to the device
        self.to(device)

    def forward(self, x):
        """ Feed forward implementation for this network """
        x = self.start_block(x)
        for resblock in self.backbone:
            x = resblock(x)
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value


class ResBlock(nn.Module):
    """ ResNet block """

    def __init__(self, num_hidden):
        """ Initialize the block """
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        """ Feed forward implementation for this block """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x