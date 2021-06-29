import matplotlib.pyplot as plt


rewards_train_engine_failure = []
episodes_train_engine_failure = []
count = 0
rewards_train_vanilla = []
episodes_train_vanilla = []
rewards_eval_vanilla = []
episodes_eval_vanilla = []

with open('./results2/train_engine_failure.txt') as f:
    for line in f:
        line = line.split('=')
        # print(float(line[3].strip()))
        rewards_train_engine_failure.append(float(line[3].strip()))
        episodes_train_engine_failure.append(count)
        count += 1

count = 0
with open('./results2/train_vanilla.txt') as f:
    for line in f:
        line = line.split('=')
        # print(float(line[3].strip()))
        rewards_train_vanilla.append(float(line[3].strip()))
        episodes_train_vanilla.append(count)
        count += 1

count = 0
with open('./results2/eval_vanilla.txt') as f:
    for line in f:
        line = line.split('=')
        # print(float(line[3].strip()))
        rewards_eval_vanilla.append(float(line[3].strip()))
        episodes_eval_vanilla.append(count)
        count += 1


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Comparison under engine failure")
plt.plot(episodes_train_engine_failure, rewards_train_engine_failure,
         'b', label='Trained DQN with engine failure')
plt.plot(episodes_train_vanilla, rewards_train_vanilla,
         'g', label='Trained DQN without engine failure')
plt.plot(episodes_eval_vanilla, rewards_eval_vanilla,
         'r', label='Vanilla DQN')
ax.set_xlabel("Iterations")
ax.set_ylabel("Average Reward")
ax.legend(loc='best')
plt.savefig('dqn_plot_engine_failure2.jpg')
plt.show()
