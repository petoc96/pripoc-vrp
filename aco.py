import numpy as np
import random
import matplotlib.pyplot as plt
from model.ant import Ant
import time


class ACO:
    def __init__(self, graph, ants_num=10, max_iter=200, beta=2, q0=0.1, show_figure=True):
        super()
        # graph - node location, service time information
        self.graph = graph
        self.ants_num = ants_num
        self.max_iter = max_iter
        self.max_load = graph.vehicle_capacity

        self.beta = beta
        self.q0 = q0

        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

    def run(self):
        start_time_total = time.time()
        # maximum number of iterations
        start_iteration = 0
        for iter in range(self.max_iter):
            # set the current vehicle load, current travel distance, and current time for each ant
            ants = list(Ant(self.graph) for _ in range(self.ants_num))
            for k in range(self.ants_num):
                # ant needs to visit all customers
                while not ants[k].index_to_visit_empty():
                    next_index = self.select_next_index(ants[k])
                    # check if the condition is satisfied after joining the position, if not, select again
                    if not ants[k].check_condition(next_index):
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0
                    # update ant path
                    ants[k].move_to_next_index(next_index)
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)
                # return to position 0
                ants[k].move_to_next_index(0)
                self.graph.local_update_pheromone(ants[k].current_index, 0)
            # calculate the routes length of all ants
            paths_distance = np.array([ant.total_travel_distance for ant in ants])
            # save the current best path
            best_index = np.argmin(paths_distance)
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                self.best_path = ants[int(best_index)].travel_path
                self.best_path_distance = paths_distance[best_index]
                self.best_vehicle_num = self.best_path.count(0) - 1
                start_iteration = iter

                print('[iteration %d]: found better path, distance: %f' % (iter, self.best_path_distance))

            # update pheromone table
            self.graph.global_update_pheromone(self.best_path, self.best_path_distance)
            given_iteration = 100
            if iter - start_iteration > given_iteration:
                print('Cannot find better solution after %d iteration' % given_iteration)
                break

        print('Final best path dist is %f, number of vehicle is %d' % (self.best_path_distance, self.best_vehicle_num))
        print('Running time: %0.3f seconds' % (time.time() - start_time_total))

        # x = np.array([i.x for i in self.graph.nodes])
        # y = np.array([i.y for i in self.graph.nodes])
        #
        # size = len(self.best_path)
        # idx_list = [idx + 1 for idx, val in enumerate(self.best_path) if val == 0]
        # res = [self.best_path[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
        # new_res = []
        # for r in res:
        #     temp = np.concatenate((res[0], r))
        #     new_res.append(temp)
        # new_res = new_res[1:]
        #
        # plt.figure()
        # plt.scatter(x, y)
        # for i in new_res:
        #     plt.plot(x[i], y[i])
        # # plt.plot(x[self.best_path], y[self.best_path])
        # plt.show()
        # print(self.best_path)

        print_graph(self.graph, self.best_path, self.best_path_distance, self.best_vehicle_num)

    def select_next_index(self, ant):
        current_index = ant.current_index
        index_to_visit = ant.index_to_visit

        transition_prob = self.graph.pheromone_mat[current_index][index_to_visit] * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            max_prob_index = np.argmax(transition_prob)
            next_index = index_to_visit[max_prob_index]
        else:
            # roulette algorithm
            next_index = stochastic_accept(index_to_visit, transition_prob)
        return next_index


def stochastic_accept(index_to_visit, transition_prob):
    # calculate N and max fitness value
    N = len(index_to_visit)

    # normalize
    sum_tran_prob = np.sum(transition_prob)
    norm_transition_prob = transition_prob/sum_tran_prob

    # select: O(1)
    while True:
        # randomly select an individual with uniform probability
        ind = int(N * random.random())
        if random.random() <= norm_transition_prob[ind]:
            return index_to_visit[ind]


def print_graph(graph, best_path, distance, vehicle_num):
    x = np.array([i.x for i in graph.nodes])
    y = np.array([i.y for i in graph.nodes])

    size = len(best_path)
    idx_list = [idx + 1 for idx, val in enumerate(best_path) if val == 0]
    res = [best_path[i: j] for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]

    new_res = []
    for r in res:
        temp = np.concatenate((res[0], r))
        new_res.append(temp)
    new_res = new_res[1:]

    plt.figure()
    plt.scatter(x, y)
    for i in new_res:
        plt.plot(x[i], y[i])
    # plt.plot(x[self.best_path], y[self.best_path])
    plt.title("Distance: " + str(distance) + " Num of vehicles: " + str(vehicle_num))
    plt.show()