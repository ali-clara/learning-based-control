import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from gridWorld import gridWorld

## ------------------------ global variables ------------------------ ##

env = gridWorld()

# (x,y) state location to Q matrix mapping
pos_to_state = np.zeros((env.grid_size[1], env.grid_size[0]), dtype=int)
k = 0
for i in range(env.grid_size[0]):
    for j in range(env.grid_size[1]):
        pos_to_state[j,i] = k
        k+=1

## ------------------------ helper functions ------------------------ ##

def choose_action(Q_matrix, state, epsilon=0.1):
    """Returns integer and string of the chosen action.
    Choses the next action to take with some randomness.
    Takes a random action ε% of the time"""

    # ε% of the time, choose a random action (0, 1, 2, 3)
    if np.random.rand() < epsilon:
        action = np.random.randint(0,4)
    # otherwise, choose the action that corresponds to the highest Q value for that state
    else:
        action = np.argmax(Q_matrix[state, :])

    return action, env.actions[action]

def update_Q_sarsa(Q_matrix, state, action, next_state, next_action, reward, alpha=0.5, gamma=0.8):
    current_q = Q_matrix[state, action]
    next_q = Q_matrix[next_state, next_action]

    Q_matrix[state, action] = current_q + alpha*(reward + gamma*next_q - current_q)
    return Q_matrix

def update_Q_qlearn(Q_matrix, state, action, next_state, reward, alpha=0.5, gamma=0.8):
    current_q = Q_matrix[state, action]
    max_next_q = max([Q_matrix[next_state, i] for i in range(len(env.actions))])

    Q_matrix[state, action] = current_q + alpha*(reward + gamma*max_next_q - current_q)
    return Q_matrix


def visualizer(Q_matrix, title):
    plt.figure(figsize=(7,4))

    x = np.arange(0, 11, 1)
    y = np.arange(0, 6, 1)

    # make the grid lines
    hlines = np.column_stack(np.broadcast_arrays(x[0], y, x[-1], y))
    vlines = np.column_stack(np.broadcast_arrays(x, y[0], x, y[-1]))
    lines = np.concatenate([hlines, vlines]).reshape(-1, 2, 2)
    line_collection = LineCollection(lines, color="black", linewidths=1, alpha=0.5)

    # print a star in the goal square
    plt.plot(env.door[0]+0.5, env.door[1]+0.5, "r*", markersize=25, alpha=0.5, label="door")
    
    # find the best action to take in each square and its index
    Q_best = np.argmax(Q_matrix, axis=1)
    best_vals = [row[Q_best[i]] for i, row in enumerate(Q_matrix)]
    Q_best = np.reshape(Q_best, (5,10))
    best_vals = np.reshape(best_vals, (5,10))

    # write the best action in the grid square
    for i, row in enumerate(Q_best):
        y_pos = i+0.5
        for j, element in enumerate(row):
            if best_vals[i,j] == 0:
                plt.text(j+0.25, y_pos, "0")
            else:   
                plt.text(j+0.25, y_pos, env.actions[element])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)

    ax = plt.gca()
    ax.add_collection(line_collection)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])

    plt.tight_layout()
    plt.show()

## ------------------------ algorithms ------------------------ ##
def do_learning(algorithm_type, Q_matrix, epochs, alpha, gamma, moving_door=False):
    epoch_reward = []
    for learning_epoch in range(epochs):
        # reset the environment to the original configuration every episode
        grid_pos = env.reset()       
        # choose an initial state-action pair                   
        state = pos_to_state[grid_pos[0], grid_pos[1]]
        action, action_str = choose_action(Q_matrix, state)

        time_step_reward = 0
        for time_step in range(20):
            # take the chosen action and return the associated reward and new state
            next_grid_pos, reward = env.step(action_str, rng_door=moving_door)
            next_state = pos_to_state[next_grid_pos[0], next_grid_pos[1]]
            # do that again
            next_action, next_action_str = choose_action(Q_matrix, next_state)
            # use the current state and next state to update the Q matrix
            if algorithm_type == "sarsa":
                Q_matrix = update_Q_sarsa(Q_matrix, state, action, next_state, next_action, reward, alpha, gamma)
            elif algorithm_type == "qlearn":
                Q_matrix = update_Q_qlearn(Q_matrix, state, action, next_state, reward, alpha, gamma)
            else:
                print("probably misspelled the algorithm")
                return
            # update loop params
            state = next_state
            action = next_action
            action_str = next_action_str

            time_step_reward+=reward

        epoch_reward.append(time_step_reward)

    return Q_matrix, epoch_reward

## ------------------------ flight code ------------------------ ##

alg = "sarsa"
reward_list = []
trials = 1
epochs = 1000
moving_door = True
alpha = 0.5
gamma = 0.7
for i in range(trials):
    # 50 states and 4 actions make a 50x4 Q matrix initialized with zeros
    Q_matrix = np.zeros((env.grid_size[0]*env.grid_size[1], len(env.actions)))

    Q_matrix_final, reward = do_learning(alg, Q_matrix, epochs, alpha, gamma, moving_door)

    reward_per_epoch = np.sum(reward)/epochs
    reward_list.append(reward_per_epoch)

avg = np.mean(reward_list)
std = np.std(reward_list)
print(f"total reward per epoch, {epochs} epochs, {trials} trials, {alg} = {avg} ± {std}")

visualizer(Q_matrix_final, alg)
# print(Q_matrix_final)

fig, ax = plt.subplots(1,1)
ax.plot(reward)
ax.set_xlabel("Epochs")
ax.set_ylabel("Reward after 20 steps")
ax.set_title(f"{alg}, α {alpha}, γ {gamma}")
plt.show()
