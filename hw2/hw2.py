import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq
from copy import copy
import time

class TSP:
    def __init__(self) -> None:
        
        self.ycoords = None
        self.xcoords = None
        
        self.dist_matrix = np.zeros((25, 25))
        self.load_cities()

        initial_path = self.initialize_path()
        self.initial_path = initial_path

        self.best_cost = 100
        self.best_state = initial_path
    
    def euc_distance(self, x1, y1, x2, y2):
        pt1 = np.array([x1, y1])
        pt2 = np.array([x2, y2])
        dist = np.linalg.norm(pt1 - pt2)
        return dist
    
    def initialize_path(self):
        cities = np.arange(0, len(self.dist_matrix[0]), 1)
        np.random.shuffle(cities)
        return cities

    def initialize_k_paths(self, k):
        initial_states = []
        for _ in range(k):
            state = self.initialize_path()
            initial_states.append(state)

        return initial_states

    def load_cities(self):
        labels = ["x", "y"]
        coords = pd.read_csv("hw2.csv", names=labels)
        x_list = coords["x"].tolist()
        y_list = coords["y"].tolist()

        self.xcoords = x_list
        self.ycoords = y_list

        for i, x in enumerate(x_list):
            y = y_list[i]
            for j, _ in enumerate(x_list):
                distance = self.euc_distance(x, y, x_list[j], y_list[j])
                self.dist_matrix[i,j] = distance

    def calculate_cost(self, city_path):
        cost = 0
        for i, city in enumerate(city_path[0:-1]):
            dist = self.dist_matrix[city_path[i], city_path[i+1]]
            # print(f"distance between city {city_path[i]} and city {city_path[i+1]}: {dist}")
            cost += dist

        # for the last point
        dist = self.dist_matrix[city_path[-1], city_path[0]]
        # print(f"distance between city {city_path[-1]} and city {city_path[0]}: {dist}")
        cost += dist
        return cost
    
    def plot_path(self, city_path, cost, title):
        fig, ax = plt.subplots(1,1)
        for i, point in enumerate(city_path[0:-1]):
            xvals = [self.xcoords[point], self.xcoords[city_path[i+1]]]
            yvals = [self.ycoords[point], self.ycoords[city_path[i+1]]]
            ax.plot(xvals, yvals, 'o', linestyle='-')
            if i == 0:
                ax.plot(xvals[0], yvals[0], '*', markersize=13, zorder=3, label="Start")
                ax.arrow(xvals[0], yvals[0], xvals[1]-xvals[0], yvals[1]-yvals[0], head_width=0.01, zorder=3)

        last_xvals = [self.xcoords[city_path[-1]], self.xcoords[city_path[0]]]
        last_yvals = [self.ycoords[city_path[-1]], self.ycoords[city_path[0]]]

        ax.plot(last_xvals, last_yvals, 'o', linestyle='-')
        ax.set_title(title+", cost "+str(round(cost,2)))
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        plt.legend(loc=0)
        plt.show()
    
    def generate_neighbor(self, city_path):
        """Uses tuple unpacking to swap two random values in the original path"""
        path = copy(city_path)
        swap1 = np.random.randint(0, 25)
        swap2 = np.random.randint(0, 25)
        while swap1 == swap2:
            swap2 = np.random.randint(0, 25)

        path[swap1], path[swap2] = path[swap2], path[swap1]

        return path
    
    def simulated_annealing(self, num_iterations, plot=True):
        temp = 500
        cost_list = []

        for i in tqdm(range(num_iterations), desc="Running Simulated Annealing"):
            temp *= 0.99
            state = self.best_state
            neighbor = self.generate_neighbor(state)
            cost = self.calculate_cost(neighbor)
            # if we've found a solution with a lower cost, take it
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_state = neighbor
            # if not, take it with probability p
            else:
                del_err = cost - self.best_cost
                p = np.exp(-del_err / temp)
                if np.random.rand() < p:
                    self.best_state = neighbor
                    self.best_cost = cost

            cost_list.append(self.best_cost)

        # plot the best path generated at the end of the iterations
        if plot is True:
            self.plot_path(self.best_state, cost=self.best_cost, title="Simulated Annealing")
        return cost_list

    def mutate(self, city_path):
        """Mutates a state for an evolutaionary algorithm by randomly swapping
        two adjacent cities, as opposed to two random cities (as in self.generate_neighbor())"""
        path = copy(city_path)
        swap1 = np.random.randint(0, 25)
        swap2 = swap1 + 1
        if swap2 > 24:
            swap2 = 0

        path[swap1], path[swap2] = path[swap2], path[swap1]

        return path
    
    def ev_alg(self, num_iterations, k, plot=True):        
        states = self.initialize_k_paths(k) # list of initial random states, length k
        cost_list = []
        for _ in tqdm(range(num_iterations), desc="Running Evolutionary Algorithm"):
            neighbors = [self.generate_neighbor(state) for state in states] # list of neighbors, length k
            mutated_neighbors = [self.mutate(neighbor) for neighbor in neighbors] # list of mutated neighbors, length k

            total_states = states + mutated_neighbors # concatenated list, length 2k
            costs = [self.calculate_cost(state) for state in total_states] # costs of all states, length 2k
            
            # select the k smallest costs and their associated states
            min_cost_indices = [costs.index(i) for i in heapq.nsmallest(k, costs)]
            states = [total_states[i] for i in min_cost_indices]
            min_costs = [costs[i] for i in min_cost_indices]
            
            cost_list.append(np.mean(min_costs))

        # plot the best path generated at the end of the iterations
        if plot is True:
            self.plot_path(states[0], cost=min_costs[0], title="Evolutionary Algorithm")
        
        return cost_list

    def create_bins(self, costs):
        costs = 1/np.array(costs)
        bins = np.cumsum(costs)
        return bins
 
    def beam_search(self, num_iterations, k, plot=True):
        states = self.initialize_k_paths(k) # list of initial random states, length k
        cost_list = []

        for _ in tqdm(range(num_iterations), desc="Running Stochastic Beam Search"):
            # generate n*k ("all") successors
            neighbors = []
            for _ in range(10):
                ns = [self.generate_neighbor(state) for state in states]
                neighbors += ns
            
            totals = states + neighbors
            costs = [self.calculate_cost(state) for state in totals] # costs of all neigbors, length nk
            
            # stochastic: randomly select k of the successors, with the value of state affecting selection probability --
                # couldn't get this to converge
            # bins = self.create_bins(costs)
            # chosen_neighbors = []
            # chosen_costs = []
            # while len(chosen_neighbors) < k:
            #     p = np.random.uniform(0, bins[-1])
            #     indices = np.where(bins > p)[0]
            #     bin = indices[0]

            #     if costs[bin] not in chosen_costs:
            #         chosen_neighbors.append(neighbors[bin])
            #         chosen_costs.append(costs[bin])

            # picking the k smallest costs
            min_cost_indices = [costs.index(i) for i in heapq.nsmallest(k, costs)]
            states = [totals[i] for i in min_cost_indices]
            min_costs = [costs[i] for i in min_cost_indices]

            cost_list.append(np.mean(min_costs))

        # plot the best path generated at the end of the iterations
        if plot is True:
            self.plot_path(states[0], cost=min_costs[0], title="Beam Search")

        return cost_list


if __name__ == "__main__":
    my_tsp = TSP()

    alg_time = []
    alg_final_cost = []
    sa_cost_list = []
    for i in range(10):
        start = time.time()
        sim_anneal_cost = my_tsp.simulated_annealing(num_iterations=5000, plot=False)
        stop = time.time()
        alg_time.append(stop-start)
        alg_final_cost.append(sim_anneal_cost[-1])
        sa_cost_list.append(sim_anneal_cost)

    print("Simulated annealing")
    print(f"Average time: {np.mean(alg_time)} s ± {np.std(alg_time)}")
    print(f"Final path cost: {np.mean(alg_final_cost)} ± {np.std(alg_final_cost)}")

    ea_k = 10
    alg_time = []
    alg_final_cost = []
    ev_alg_cost_list = []
    for i in range(10):
        start = time.time()
        ev_alg_cost = my_tsp.ev_alg(num_iterations=5000, k=ea_k, plot=False)
        stop = time.time()
        alg_time.append(stop-start)
        alg_final_cost.append(ev_alg_cost[-1])
        ev_alg_cost_list.append(ev_alg_cost)

    print(f"Evolutionary algorithm")
    print(f"Average time: {np.mean(alg_time)} s ± {np.std(alg_time)}")
    print(f"Final path cost: {np.mean(alg_final_cost)} ± {np.std(alg_final_cost)}")

    beam_k = 5
    alg_time = []
    alg_final_cost = []
    beam_cost_list = []
    for i in range(10):
        start = time.time()
        stoch_beam_cost = my_tsp.beam_search(num_iterations=5000, k=beam_k, plot=False)
        stop=time.time()
        alg_time.append(stop-start)
        alg_final_cost.append(stoch_beam_cost[-1])
        beam_cost_list.append(stoch_beam_cost)

    print(f"Beam search")
    print(f"Average time: {np.mean(alg_time)} s ± {np.std(alg_time)}")
    print(f"Final path cost: {np.mean(alg_final_cost)} ± {np.std(alg_final_cost)}")

    fig, ax = plt.subplots(1,1)
    ax.plot(np.mean(sa_cost_list, axis=0), label="Simulated Annealing")
    ax.plot(np.mean(ev_alg_cost_list, axis=0), label=f"Evolutionary Algorithm, k={ea_k}")
    ax.plot(np.mean(beam_cost_list, axis=0), label=f"Beam Search, k={beam_k}", alpha=0.7)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Path Cost")
    ax.set_title("Average path cost of 3 algorithms")
    plt.legend(loc=0)
    plt.show()
