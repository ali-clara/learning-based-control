import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from hw4_network import NeuralNetwork

# ## ------- set up matplotlib for live plotting ------- ##
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
# plt.ion()

## ------- set up environment ------- ## 
env = gym.make("CartPole-v1")
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

## ------- helper functions ------- ##

def select_action(network, state, epsilon=0.1):
    prob = np.random.uniform()
    if prob < epsilon:
        # random
        action = env.action_space.sample()
    else:
        action = np.round(network.forward_pass(state, test=True))
        action = int(action)

    return action

def generate_networks(num_networks):
    network_list = np.zeros((num_networks, 2), dtype=object)
    for i, _ in enumerate(network_list):
        network_list[i,0] = NeuralNetwork(num_inputs=n_observations, num_outputs=1)

    return network_list

def select_network(network_list, num_returned, epsilon=0.1, test=False):
    """Either selects the n best networks from the list or choses n random networks from the list"""
    prob = np.random.uniform()
    # ε% of the time, choose a random network
    if prob < epsilon:
        networks = np.random.choice(network_list[:,0], size=num_returned, replace=False)
    # otherwise, choose the network with the highest fitness (located in network_list[:,1])
    else:
        # sort the fitnesses of each network
        sorted_network_indices = np.argsort(network_list[:,1])
        # invert that so the highest fitness is at the top
        sorted_network_indices = sorted_network_indices[::-1]
        # pick the n best networks
        n_indices = sorted_network_indices[0:num_returned]
        best_n_networks = network_list[n_indices]
        networks = best_n_networks[:,0]

    if test:
        return networks
    else:
        return copy.deepcopy(networks)

def remove_network(network_list, num_removed):
    """Removes the n worst networks from the list"""
    # sort the fitnesses of each network
    sorted_network_indices = np.argsort(network_list[:,1])
    # pick the n worst networks
    n_indices = sorted_network_indices[0:num_removed]
    network_list = np.delete(network_list, n_indices, axis=0)

    return network_list

def swap_weight_layer(weight1, weight2):
    layer = np.random.randint(0, weight1.shape[0])
    weight1[layer] = weight2[layer]
    return weight1

def swap_one_weight(weight1, weight2):
    swap1 = np.random.randint(0, weight1.shape[0])
    swap2 = np.random.randint(0, weight1.shape[1])
    weight1[swap1, swap2] = weight2[swap1, swap2]
    return weight1

def crossover(parents):
    """Swaps either a weight (90% of the time) or a layer of weights (10% of the time) 
        between two networks to create a child
        Inputs: two parent neural networks
        Output: child neural network"""
    # initialize a child
    parent1, parent2 = parents
    child = copy.deepcopy(parent1)
    # randomly select between changing w1 or w2
    w = np.random.randint(0,2)
    # randomly select between swapping one weight or a layer of weights
    prob = np.random.uniform()

    # do the swapping
    if w == 0:
        if prob < 0.1:
            weight = swap_weight_layer(parent1.w1, parent2.w1) 
            child.w1 = weight
        else:
            weight = swap_one_weight(parent1.w1, parent2.w1)
            child.w1 = weight
    else:
        if prob < 0.1:
            weight = swap_weight_layer(parent1.w2, parent2.w2) 
            child.w2 = weight
        else:
            weight = swap_one_weight(parent1.w2, parent2.w2)
            child.w2 = weight

    return child

def mutate(network):
    """Mutates a network by randomly choosing one of four mutations:
        1. completely replacing with a new random value
        2. changing by some percentage (50% to 150%)
        3. adding a random number between -1 and 1
        4. changing the sign of the weight
        """
    # randomly select between changing w1 or w2
    w = np.random.randint(0,2)
    weight_matrices = [network.w1, network.w2]
    weight_matrix = weight_matrices[w]
    
    # pick a random weight to change
    pos1 = np.random.randint(0, weight_matrix.shape[0])
    pos2 = np.random.randint(0, weight_matrix.shape[1])

    prob = np.random.uniform()
    # replace with new random value
    if prob <= 0.25:
        new_rand_val = np.random.uniform()
        weight_matrix[pos1, pos2] = new_rand_val
    # change by some percentage
    elif prob > 0.25 and prob <= 0.5:
        scale = np.random.uniform(0.5, 1.5)
        weight_matrix[pos1, pos2] *= scale
    # add random number
    elif prob > 0.5 and prob <= 0.75:
        rand_addition = np.random.uniform(-1, 1)
        weight_matrix[pos1, pos2] += rand_addition
    # change sign
    else:
        weight_matrix[pos1, pos2] *= -1

    weight_matrices[w] = weight_matrix
    return network

## ------- learning algorithm ------- ##

def initialize_networks(num_networks):
    """Runs each network once to initialize their fitnesses"""
    network_list = generate_networks(num_networks)
    for i, val in enumerate(network_list):
        # "val" is [network, fitness]
        network = val[0]

        state, info = env.reset()
        action = select_action(network, state)

        time_step_reward = 0
        for time_step in range(500):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = select_action(network, next_state)

            # update loop params
            state = next_state
            action = next_action

            time_step_reward += reward

            done = terminated or truncated
            if done:
                break

        network_list[i,1] = time_step_reward

    return network_list
        
def do_learning(epochs, num_networks):
    # initialize N neural networks
    network_list = initialize_networks(num_networks)
    
    best_epoch_reward = []
    for learning_epoch in tqdm(range(epochs)):
        # pick network using epsilon-greedy alg
        network1, network2 = select_network(network_list, 2)
        # create child out of networks
        child = crossover([network1, network2])
        # randomly modify network parameters
        network1 = mutate(network1)
        network2 = mutate(network2)
        child = mutate(child)

        # use network on agent for T steps
        for network in [network1, network2, child]:
            state, info = env.reset()
            action = select_action(network, state)

            n_network_time_steps = []
            time_step_reward = 0
            for time_step in range(500):
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = select_action(network, next_state)

                # update loop params
                state = next_state
                action = next_action

                time_step_reward += reward

                done = terminated or truncated
                if done:
                    break

            # evaluate network performance
            # reinsert into pool
            network_list = np.append(network_list, [[network, time_step_reward]], axis=0)
            # remove worst networks from pool
            network_list = remove_network(network_list, 1)
            # print(network_list)
            # print("--")
            n_network_time_steps.append(time_step_reward)
        
        
        # best_epoch_reward.append(max(n_network_time_steps))
        best_epoch_reward.append(max(network_list[:,1]))

        # repeat

    return best_epoch_reward, network_list

def test():
    env = gym.make("CartPole-v1", render_mode="human")

    w1 = np.load("best_evo_weights/w1.npy")
    w2 = np.load("best_evo_weights/w2.npy")
    b1 = np.load("best_evo_weights/b1.npy")
    b2 = np.load("best_evo_weights/b2.npy")

    network = NeuralNetwork(num_inputs=n_observations, num_outputs=1, weights=[w1,w2,b1,b2])
    state, info = env.reset()
    action = select_action(network, state, epsilon=0)
    test_reward = 0
    for time_step in range(500):
        env.render()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_action = select_action(network, next_state, epsilon=0)

        # update loop params
        state = next_state
        action = next_action
        test_reward += reward

        if terminated or truncated:
            print(f"terminated after {test_reward} steps")
            break

    return test_reward

## ------- flight code ------- ##

epochs = 1000
num_networks = 10

train = False

if train:
    best_reward, network_list = do_learning(epochs, num_networks)
    print(network_list)
    fig, ax = plt.subplots(1,1)
    ax.plot(best_reward)
    ax.set_xlabel("epochs")
    ax.set_ylabel("time steps kept upright")
    ax.set_title("Cart Pole Reward")
    plt.show()

    save = input("Save weights? (y/n) \n")
    if save == "y":
        best_network = select_network(network_list, 1, test=True, epsilon=0)
        print(best_network)
        best_network[0].save_weights()

else:
    test()
    

# reward_per_epoch = np.sum(reward)/epochs
# print(f"total reward per {epochs} epochs: {reward_per_epoch}"s)

# fig, ax = plt.subplots(1,1)
# ax.plot(reward)
# ax.set_xlabel("epochs")
# ax.set_ylabel("time steps kept upright")
# ax.set_title("Cart Pole Reward")

# plt.show()






