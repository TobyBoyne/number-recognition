import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_target(self, labels):
        target = torch.zeros([len(labels), 10])
        for i, n in enumerate(labels):
            target[i, n] = 1.0
        return target

    # def train_loop(self, inputs, labels):
    #     self.optim.zero_grad()  # zero the gradient buffers
    #     output = net(x_in)
    #     target = self.get_target(target_n)
    #     loss = loss_fn(output, target)
    #     loss.backward()
    #     self.optim.step()



if __name__ == "__main__":
    net = Net()
    print(net)

    x_in = torch.randn(1, 1, 28, 28)
    net.train_loop(x_in, 1)
    print(net.conv2.bias.grad)