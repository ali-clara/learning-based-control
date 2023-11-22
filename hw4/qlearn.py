import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hw4_network import NeuralNetwork

## ------- set up environment ------- ## 
env = gym.make("CartPole-v1")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

## ------- set up Q-learning neural network ------- ##

# inputs: environment state \\ outputs: Q(s,a) values for the state
Q_network = NeuralNetwork(num_inputs=n_observations, num_outputs=n_actions) 

## ------- helper functions ------- ##

def select_action(state, epsilon=0.1):
    prob = np.random.uniform()
    if prob < epsilon:
        # random
        action = env.action_space.sample()
    else:
        # action with the best Q-value, according to the network
        q_vals = Q_network.forward_pass(state, test=True)
        action = np.argmax(q_vals)

    return action

def update_Q_network(state, action, next_state, reward, alpha=0.5, gamma=0.8):
    # query the neural network
    z1, z2, a1, q_vals = Q_network.forward_pass(state, test=False)
    # grab the Q value corresponding to the current state and action
    current_q = q_vals[0][action]
    # find the max Q value for the next state
    max_next_q = np.max(Q_network.forward_pass(next_state, test=True))
    # update the network
    current_q_update = current_q + alpha*(reward + gamma*max_next_q - current_q)
    dE_dy = Q_network.d_error(current_q_update, current_q)
    Q_network.backward_pass(state, dE_dy, a1, z1, z2)

def test():
    # env = gym.make("CartPole-v1", render_mode="human")

    # w1 = np.load("best_qlearn_weights/w1.npy")
    # w2 = np.load("best_qlearn_weights/w2.npy")
    # b1 = np.load("best_qlearn_weights/b1.npy")
    # b2 = np.load("best_qlearn_weights/b2.npy")

    w1 = np.load("w1.npy")
    w2 = np.load("w2.npy")
    b1 = np.load("b1.npy")
    b2 = np.load("b2.npy")

    network = NeuralNetwork(num_inputs=n_observations, num_outputs=1, weights=[w1,w2,b1,b2])
    state, info = env.reset()
    action = select_action(state, epsilon=0)
    test_reward = 0
    for time_step in range(500):
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = select_action(next_state, epsilon=0)

        # update loop params
        state = next_state
        action = next_action
        test_reward += reward

        if terminated or truncated:
            print(f"terminated after {test_reward} steps")
            break

    return test_reward

## ------- learning algorithm ------- ##

def do_learning(epochs, alpha, gamma):
    epoch_reward = []
    for learning_epoch in tqdm(range(epochs)):
        lr = alpha / (1+0.5*learning_epoch)
        state, info = env.reset()
        action = select_action(state)
        time_step_reward = 0
        for time_step in range(500):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = select_action(next_state)
            update_Q_network(state, action, next_state, reward, alpha, gamma)

            done = terminated or truncated

            # update loop params
            state = next_state
            action = next_action

            time_step_reward += reward

            if done:
                break

        epoch_reward.append(time_step_reward)

    return epoch_reward

## ------- flight code ------- ##

epochs = 1000
alpha = 0.2
gamma = 0.8

train = False

if train:
    reward = do_learning(epochs, alpha, gamma)

    reward_per_epoch = np.sum(reward)/epochs
    print(f"total reward per {epochs} epochs: {reward_per_epoch}")

    fig, ax = plt.subplots(1,1)
    ax.plot(reward)
    ax.set_xlabel("epochs")
    ax.set_ylabel("time steps kept upright")
    ax.set_title("Cart Pole Reward")

    plt.show()

    save = input("Save weights? (y/n) \n")
    if save == "y":
        Q_network.save_weights()

else:
    total_reward = []
    trials = 10
    for _ in range(trials):
        reward = test()
        total_reward.append(reward)

    print(f"Mean after {trials} trials: {np.mean(total_reward)} ± {np.std(total_reward)}")




