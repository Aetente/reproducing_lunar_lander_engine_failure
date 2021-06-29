import numpy as np
import lunar_lander as lander
from collections import deque
import gym
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


model_path = sys.argv[1]


class QNetwork(nn.Module):
    def __init__(self, state_size):
        super(QNetwork, self).__init__()

        hidden_1 = 128
        hidden_2 = 128

        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.z1 = nn.Linear(state_size, hidden_1)

        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.z2 = nn.Linear(hidden_1, hidden_2)

        self.z3 = nn.Linear(hidden_2, 4)

    def forward(self, h):
        h = F.relu(self.z1(h))

        h = F.relu(self.z2(h))

        return self.z3(h)


model = QNetwork(state_size=8)
model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    env = lander.LunarLander()
    num_episodes = 400
    np.random.seed(0)
    scores = []
    model.eval()
    for i in range(num_episodes+1):
        score = 0
        state = env.reset()
        finished = False
        for j in range(3000):
            state = np.reshape(state, (1, 8))
            action_values = model(torch.from_numpy(state))
            _, action = torch.max(action_values.detach(), 1)
            action = action.detach().numpy()[0]
            # env.render()
            if np.random.random() > 0.2:
                next_state, reward, finished, metadata = env.step(action)
            else:
                next_state, reward, finished, metadata = env.step(0)
            next_state = np.reshape(next_state, (1, 8))
            score += reward
            state = next_state
            if finished:
                scores.append(score)
                print("Episode = {}, Score = {}, Avg_Score = {}".format(
                    i, score, np.mean(scores[-100:])))
                break
