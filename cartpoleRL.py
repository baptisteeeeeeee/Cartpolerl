import gym
import math, random as rd, numpy as np, copy, matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.functional as F
import torch.optim as omptim


env = gym.make('CartPole-v1', render_mode='human')
obs = env.reset()


gamma = 0.95
lr = 0.0001
epsilon, epmax, epmin, epdecay = 1, 1, 0.1, 0.005
N_episodes = 3000



n_input, n_hidden, n_out = 4, 5, 2

dqn = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.Tanh(),
                      nn.Linear(n_hidden, n_hidden),
                      nn.Tanh(),
                      nn.Linear(n_hidden, n_out))
loss_function = nn.MSELoss()
optimizer = omptim.SGD(dqn.parameters(), lr)
target_network = copy.deepcopy(dqn)
X = 10
cX = 0


replay_memory = [[], [], [], []]

def training_data():
    memory_size = len(replay_memory[1])
    k = rd.randint(2,3)
    indices = [int(x) for x in np.random.choice(memory_size, int(memory_size/k), replace=True)]
    return [np.array(replay_memory[0])[indices].tolist(), np.array(replay_memory[1])[indices].tolist(), np.array(replay_memory[2])[indices].tolist(), np.array(replay_memory[3])[indices].tolist()]


def update_epsilon(N):
    global epsilon
    epsilon = epmin + (epmax - epmin) * math.exp(-epdecay * N)


def getAction(state, noExploration = False):
    if not noExploration:
        f = rd.random()
        if f < epsilon:
            return rd.randint(0,1)
    q_values = dqn(T.tensor(np.array([state])).to(T.float32))
    if q_values[0][0].item() > q_values[0][1].item() :
        return 0
    return 1

def equals(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True   

rs = []

for e in range(N_episodes):
    current_state, d1, d2 = env.reset()[0], False, False
    update_epsilon(e)
    replay_memory = [[], [], [], []]

    tr= 0

    print('Episode ', e, ':   epsilon: ', epsilon, '    learning rate: ', lr)

    while d1 == False and d2 == False:
        action = getAction(current_state)

        new_state, reward, d1, d2, _ = env.step(action)

        replay_memory[0].append([*current_state])
        replay_memory[1].append(action)
        replay_memory[2].append(reward)
        replay_memory[3].append([*new_state])

        current_state = new_state

        tr+=reward
    rs.append(tr)


    tset = training_data()
    tset_size = np.arange(len(tset[1]))

    state_t = T.tensor(np.array(tset[0])).to(T.float32)
    state_t1 = T.tensor(np.array(tset[3])).to(T.float32)
    rewards = T.tensor(np.array(tset[2])).to(T.float32)

    q_values = dqn(state_t)
    next_q_values = target_network(state_t1)

    predicted_values = q_values[tset_size, tset[1][action]]
    target_q_values = rewards + gamma * T.max(next_q_values, dim=1)[0]

    loss = loss_function(target_q_values, predicted_values)

    dqn.zero_grad()
    loss.backward()
    optimizer.step()

    cX+=1
    if cX >= X:
        cX = 0
        target_network = copy.deepcopy(dqn)
    
  
plt.plot(np.linspace(0, N_episodes-1, num=N_episodes), np.array(rs))
plt.show()