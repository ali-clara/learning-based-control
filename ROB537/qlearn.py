import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sim2D import SIM2D

## ------------------------ global variables ------------------------ ##

env = SIM2D()

# intialize q matrix to match grid world and action space dimensions, currently 10x12x5
Q_matrix = np.zeros((env.grid_world[0], env.grid_world[1], len(env.actions)))

## ------------------------ helper functions ------------------------ ##

def choose_action(Q_matrix, state, epsilon=0.1):
    """Returns integer and string of the chosen action.
    Choses the next action to take with some randomness.
    Takes a random action ε% of the time"""

    # ε% of the time, choose a random action (0, 1, 2, 3, 4)
    if np.random.rand() < epsilon:
        action = np.random.randint(0,5)
    # otherwise, choose the action that corresponds to the highest Q value for that state
    else:
        action = np.argmax(Q_matrix[state[0], state[1], :])

    return action, env.actions[action]

def update_Q_sarsa(Q_matrix, state, action, next_state, next_action, reward, alpha=0.5, gamma=0.8):
    current_q_index = state[0], state[1], action
    next_q_index = next_state[0], next_state[1], next_action

    current_q = Q_matrix[current_q_index]
    next_q = Q_matrix[next_q_index]

    Q_matrix[current_q_index] = current_q + alpha*(reward + gamma*next_q - current_q)
    return Q_matrix

def update_Q_qlearn(Q_matrix, state, action, next_state, reward, alpha=0.5, gamma=0.8):
    current_q_index = state[0], state[1], action
    current_q = Q_matrix[current_q_index]

    max_next_q = np.max(Q_matrix[next_state[0], next_state[1], :])

    Q_matrix[current_q_index] = current_q + alpha*(reward + gamma*max_next_q - current_q)
    return Q_matrix

def visualizer():
    fig, ax = env.make_plot(show=False)
    ax.grid(which='both')

    best_actions = np.argmax(Q_matrix, axis=2)

    # draw the best action in the grid square
    arrows = [[0,0.5], [0,-0.5], [0.5,0], [-0.5,0], [0,0]]
    for i, row in enumerate(best_actions):
        x_pos = i
        for j, element in enumerate(row):  
            # plt.text(x_pos, j+0.25, env.actions[element])
            y_pos = j
            if element == 4:
                ax.plot(x_pos, y_pos, 'k.', alpha=0.25)
            else:
                ax.arrow(x_pos, y_pos, arrows[element][0], arrows[element][1], head_width=0.15, length_includes_head=True, facecolor='k', alpha=0.25)

    plt.show()

## ------------------------ learning algorithm ------------------------ ##
def do_learning(algorithm_type, Q_matrix, epochs, alpha, gamma):
    epoch_reward = []
    for learning_epoch in range(epochs):
        # reset the environment to the original configuration every episode
        state = env.reset()       
        # choose an initial state-action pair                   
        action, action_str = choose_action(Q_matrix, state)

        time_step_reward = 0
        for time_step in range(20):
            # take the chosen action and return the associated reward and new state
            next_state, reward = env.step(action_str)
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

alg = "qlearn"
reward_list = []
trials = 1
epochs = 5000
alpha = 0.5
gamma = 0.7

Q_matrix_final, reward = do_learning(alg, Q_matrix, epochs, alpha, gamma)
# print(Q_matrix_final, reward)

visualizer()

plt.plot(reward)
plt.show()

# q_long = Q_matrix.reshape(env.grid_world[0]*env.grid_world[1], len(env.actions))
# print(np.argmax(q_long, axis=1))

# print(np.argmax(Q_matrix, axis=2)[1,0])

# # find the best action of the point (0,1)
# print(np.argmax(Q_matrix[0,0,:]))

# print(Q_matrix[:,:,0])

