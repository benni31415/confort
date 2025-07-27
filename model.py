from torch import nn

class DQNetwork(nn.Module):

    def __init__(self, n_observations=64, n_actions=8):
        super(DQNetwork, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4)
        # Convolution output: (8-3)x(8-3)x3
        self.linear1 = nn.Linear(75, 128, bias=False)
        self.linear2 = nn.Linear(128, n_actions, bias=False)
        self.double()

    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        x = self.linear1(x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
        return self.linear2(x)