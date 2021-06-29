import numpy as np
import lunar_lander as lander
from collections import deque
import gym
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


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


learning_rate = 0.001


epsilon = 1
gamma = .99
batch_size = 64
memory = deque(maxlen=1000000)
min_eps = 0.01


model = QNetwork(state_size=8)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def replay_experiences():
    if len(memory) >= batch_size:
        sample_choices = np.array(memory)
        mini_batch_index = np.random.choice(len(sample_choices), batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        finishes = []
        for index in mini_batch_index:
            states.append(memory[index][0])
            actions.append(memory[index][1])
            next_states.append(memory[index][2])
            rewards.append(memory[index][3])
            finishes.append(memory[index][4])
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        finishes = np.array(finishes)
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        next_states = torch.from_numpy(next_states)
        states = torch.from_numpy(states)
        q_vals_next_state = model(next_states)
        q_vals_target = model(states)
        max_q_values_next_state = torch.max(q_vals_next_state, 1)[0]
        q_vals_target = q_vals_target.detach().numpy()
        q_vals_target[np.arange(batch_size).astype(int), actions.astype(int)] = rewards + \
            gamma * (max_q_values_next_state.detach().numpy()) * (1 - finishes)
        q_vals_new = model(states)
        loss = loss_function(
            q_vals_new, torch.from_numpy(q_vals_target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global epsilon
        if epsilon > min_eps:
            epsilon *= 0.996


if __name__ == '__main__':
    env = lander.LunarLander()
    num_episodes = 400
    np.random.seed(0)
    scores = []
    model.train()
    for i in range(num_episodes+1):
        score = 0
        state = env.reset()
        finished = False
        if i != 0 and i % 50 == 0:
            torch.save(model.state_dict(),
                       ".\saved_models\model128_"+str(i)+"_episodes.pth")
        for j in range(3000):
            state = np.reshape(state, (1, 8))
            if np.random.random() <= epsilon:
                action = np.random.choice(4)
            else:
                action_values = model(torch.from_numpy(state))
                _, action = torch.max(action_values.detach(), 1)
                action = action.detach().numpy()[0]
            # env.render()
            next_state, reward, finished, metadata = env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            memory.append((state, action, next_state, reward, finished))
            replay_experiences()
            score += reward
            state = next_state
            if finished:
                scores.append(score)
                print("Episode = {}, Score = {}, Avg_Score = {}".format(
                    i, score, np.mean(scores[-100:])))
                break
