import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## ------------- INITIALIZE GRIDWORLD ------------- ##
from sim2D import SIM2D
gridworld = SIM2D()

## ------------- INITIALIZE PYTORCH ------------- ##
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.flatten = nn.Flatten()

        n_hidden = 10
        n_inputs = n_inputs
        n_outputs = 2

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

class NeuroEvolution():
    def __init__(self, num_states) -> None:

        self.num_states = num_states
        self.network_pool = None
    
    def _rand_matrix_index(self, matrix):
        """ Mutation and crossover helper function.
            Returns (list) a random valid index within a matrix"""
        dimension = matrix.shape
        indices = []
        for dim in dimension:
            indices.append(np.random.randint(0, dim))

        return indices
    
    def _select_weight_matrix(self, model_dictionaries):
        """Mutation and crossover helper function.
            Takes in a list of dictionaries representing the model weights
            and biases (called by model.state_dict()), picks a random weight or bias
            matrix, and returns a list of each dict matrix"""
        # randomly select a weight or bias matrix
        model_state_dict = model_dictionaries[0]
        keys = list(model_state_dict.keys())
        # weight_matrix_key = np.random.choice(keys)
        weight_matrix_key = keys[0]
        # grab that matrix for all given inputs
        weight_matrices = [dict[weight_matrix_key] for dict in model_dictionaries]

        return weight_matrices, weight_matrix_key

    def mutate(self, model):
        """Mutates a network by randomly choosing one of four mutations:
            1. completely replacing with a new random value
            2. changing by some percentage (50% to 150%)
            3. adding a random number between -1 and 1
            4. changing the sign of the weight
            """
        # pull random weight or bias matrix from model
        state_dict_copy = copy.copy(model.state_dict())
        weight_matrix, key = self._select_weight_matrix([state_dict_copy])
        weight_matrix = weight_matrix[0]
        
        # pick a random value to change
        pos1, pos2 = self._rand_matrix_index(weight_matrix)

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

        # load new state dictionary back into model
        state_dict_copy[key] = weight_matrix
        model.load_state_dict(state_dict_copy)
        return model
    
    def crossover(self, model1, model2):
        """Swaps either a weight (90% of the time) or a layer of weights (10% of the time) 
            between two networks to create a child
            Inputs: two parent neural networks
            Output: child neural network
            """
        # initialize a child
        child = copy.deepcopy(model1)
        # pull random weight or bias matrix for parents
        state_dict1 = copy.deepcopy(model1.state_dict())    # also the state dict for child
        state_dict2 = copy.deepcopy(model2.state_dict())
        weight_matrices, key = self._select_weight_matrix([state_dict1, state_dict2])
        weight_matrix1, weight_matrix2 = weight_matrices

        prob = np.random.uniform()
        # 10% of the time, swap a layer of weights (doesn't currently swap all bias terms)
        if prob < 0.1:
            layer = self._rand_matrix_index(weight_matrix1)[0]
            weight_matrix1[layer] = weight_matrix2[layer]
        # 90% of the time, swap one weight
        else:
            try:
                swap1, swap2 = self._rand_matrix_index(weight_matrix1)
                weight_matrix1[swap1, swap2] = weight_matrix2[swap1, swap2]
                print(swap1, swap2)
            # breaks for a 1D matrix (bias terms)
            except ValueError:
                swap = self._rand_matrix_index(weight_matrix1)[0]
                weight_matrix1[swap] = weight_matrix2[swap]
            
        state_dict1[key] = weight_matrix1 
        child.load_state_dict(state_dict1)
        return child
    
    def _generate_networks(self, num_networks):
        self.network_pool = np.zeros((num_networks, 2), dtype=object)
        for i, _ in enumerate(self.network_pool):
            self.network_pool[i,0] = NeuralNetwork(self.num_states)

    def _select_networks(self, num_returned, epsilon=0.1, test=False):
        """Either selects the n best networks from the pool or choses n random networks from the pool"""
        prob = np.random.uniform()
        # ε% of the time, choose a random network
        if prob < epsilon:
            networks = np.random.choice(self.network_pool[:,0], size=num_returned, replace=False)
        # otherwise, choose the network with the highest fitness (located in network_pool[:,1])
        else:
            # sort the fitnesses of each network
            sorted_network_indices = np.argsort(self.network_pool[:,1])
            # invert that so the highest fitness is at the top
            sorted_network_indices = sorted_network_indices[::-1]
            # pick the n best networks
            n_indices = sorted_network_indices[0:num_returned]
            best_n_networks = self.network_pool[n_indices]
            networks = best_n_networks[:,0]

        if test:
            return networks
        else:
            return copy.deepcopy(networks)
        
    def _remove_networks(self, num_removed):
        """Removes the n worst networks from the list"""
        # sort the fitnesses of each network
        sorted_network_indices = np.argsort(self.network_pool[:,1])
        # pick the n worst networks to remove
        n_indices = sorted_network_indices[0:num_removed]
        self.network_pool = np.delete(self.network_pool, n_indices, axis=0)

    def select_action(self, network, state, epsilon=0.1):
        """in the gridworld, actions are currently (11/22) move by (Δx, Δy). Going 
            to eventualy move to (θ, φ) angle pan"""
        prob = np.random.uniform()
        # ε% of the time, move randomly
        if prob < epsilon:
            action = gridworld.sample_action_continuous()
            action = torch.tensor([action], device=device, dtype=torch.float32, requires_grad=False)
        # otherwise, query the network
        else:
            action = network(state)

        return action
            
## ------------- FLIGHT CODE ------------- ##
if __name__ == "__main__":

    states = gridworld.return_state()
    input_shape = states.shape
    num_states = states.size

    model = NeuralNetwork(num_states)
    model2 = NeuralNetwork(num_states)

    input = torch.tensor(states, device=device, dtype=torch.float32, requires_grad=False)
    output = model(input)
    # print(input)

    # access the weights and biases
    # for name, param in model.named_parameters():
    #     weight = param.data # tensor
        # print(name)

    alg = NeuroEvolution(num_states)
    alg._generate_networks(3)
    alg.network_pool[1,1] = 700
    print(alg.network_pool)
    alg._remove_networks(1)
    print(alg.network_pool)

