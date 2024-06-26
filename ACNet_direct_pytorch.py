import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Parameters for training
GRAD_CLIP = 1000.0
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
RNN_SIZE = 512
GOAL_REPR_SIZE = 12

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(tensor):
        out = np.random.randn(*tensor.shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return torch.tensor(out)
    return _initializer

class ACNet(nn.Module):
    def __init__(self, scope, a_size, trainer, TRAINING, GRID_SIZE, GLOBAL_NET_SCOPE):
        super(ACNet, self).__init__()
        
        self.scope = scope
        # input should be integrated here.. Farhan
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(RNN_SIZE//4, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE//2, RNN_SIZE//2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(RNN_SIZE//2, RNN_SIZE-GOAL_REPR_SIZE, kernel_size=2, stride=1)

        # Fully connected layers
        self.fc_goal = nn.Linear(3, GOAL_REPR_SIZE)
        self.fc_hidden = nn.Linear(RNN_SIZE, RNN_SIZE)
        
        # LSTM layer
        self.lstm = nn.LSTMCell(RNN_SIZE, RNN_SIZE)
        
        # Output layers
        self.policy_layer = nn.Linear(RNN_SIZE, a_size)
        self.value_layer = nn.Linear(RNN_SIZE, 1)
        self.blocking_layer = nn.Linear(RNN_SIZE, 1)
        self.on_goal_layer = nn.Linear(RNN_SIZE, 1)

        # Apply custom weight initialization
        self.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data = normalized_columns_initializer(1.0)(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
    
    def forward(self, inputs, goal_pos, hxs, cxs):
        # print("inside forward")
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        goal_layer = F.relu(self.fc_goal(goal_pos))
        x = torch.cat([x, goal_layer], 1)
        x = F.relu(self.fc_hidden(x))
        
        hx, cx = self.lstm(x, (hxs, cxs))
        x = hx
        
        policy = F.softmax(self.policy_layer(x), dim=-1)
        value = self.value_layer(x)
        blocking = torch.sigmoid(self.blocking_layer(x))
        on_goal = torch.sigmoid(self.on_goal_layer(x))
        
        return policy, value, (hx, cx), blocking, on_goal

    def evaluate_actions(self, inputs, goal_pos, hxs, cxs, actions):
        policy, value, (hx, cx), blocking, on_goal = self.forward(inputs, goal_pos, hxs, cxs)
        log_probs = torch.log(policy.gather(1, actions.view(-1, 1)))
        entropy = -(policy * log_probs).sum(-1).mean()
        return value, log_probs, entropy, (hx, cx), blocking, on_goal

# # Usage example
# net = ACNet('scope', a_size=5, trainer=optim.Adam, TRAINING=True, GRID_SIZE=10, GLOBAL_NET_SCOPE='global')
# inputs = torch.randn(1, 4, 10, 10)
# goal_pos = torch.randn(1, 3)
# hxs = torch.zeros(1, RNN_SIZE)
# cxs = torch.zeros(1, RNN_SIZE)
# policy, value, (hx, cx), blocking, on_goal = net(inputs, goal_pos, hxs, cxs)
# print(policy, value, (hx, cx), blocking, on_goal)
