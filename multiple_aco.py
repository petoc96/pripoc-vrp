# implementacia algoritmu MACS-VRPTW
#
import numpy as np
import random
from model.graph import Graph, Path
from model.ant import Ant
from threading import Thread, Event
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import time
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import matplotlib.pyplot as plt


class MultipleAntColonySystem:
    def __init__(self, graph, ants_num=10, beta=1, q0=0.1):
        super()
        # node location, service time information
        self.graph = graph
        # number of ants
        self.ants_num = ants_num
        # represents the maximum load of each vehicle
        self.max_load = graph.vehicle_capacity
        # the importance of enlightening information
        self.beta = beta
        # represents the probability of directly selecting the next point with the highest probability
        self.q0 = q0
        # best path
        self.best_path_distance = None
        self.best_path = None
        self.best_vehicle_num = None

    @staticmethod
    def stochastic_accept(cities_to_visit, probabilities):
        # calculate N and max fitness value
        N = len(cities_to_visit)
        # normalize
        sum_probs = np.sum(probabilities)
        norm_probs = probabilities / sum_probs
        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_probs[ind]:
                return cities_to_visit[ind]

    @staticmethod
    def new_active_ant(ant, vehicle_num, local_search, IN, q0, beta, stop_event):
        # In new_active_ant, a maximum of vehicle_num vehicles can be used, that is, a maximum of vehicle_num + 1 depot
        # nodes can be included. Since one departure node uses one, only vehicle depots are left
        unused_depot_count = vehicle_num

        # if there are still unvisited nodes, and can also return to the depot
        while not ant.index_to_visit_empty() and unused_depot_count > 0:
            if stop_event.is_set():
                return

            # calculate all the next nodes that meet the load and other restrictions
            next_index_meet_constrains = ant.calculate_next_index_meet_constrains()

            # if there is no next node that meets the limit, go back to the depot
            if len(next_index_meet_constrains) == 0:
                ant.move_to_next_index(0)
                unused_depot_count -= 1
                continue

            # start calculating the next node that meets the limit, and select the probability of each node
            length = len(next_index_meet_constrains)
            ready_time = np.zeros(length)
            due_time = np.zeros(length)

            for i in range(length):
                ready_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].ready_time
                due_time[i] = ant.graph.nodes[next_index_meet_constrains[i]].due_time

            delivery_time = np.maximum(
                ant.vehicle_travel_time + ant.graph.node_dist_mat[ant.current_index][next_index_meet_constrains],
                ready_time)
            delta_time = delivery_time - ant.vehicle_travel_time
            distance = delta_time * (due_time - ant.vehicle_travel_time)

            distance = np.maximum(1.0, distance - IN[next_index_meet_constrains])
            closeness = 1 / distance

            transition_prob = ant.graph.pheromone_mat[ant.current_index][next_index_meet_constrains] * \
                              np.power(closeness, beta)
            transition_prob = transition_prob / np.sum(transition_prob)

            # select the node with the largest closeness according to probability
            if np.random.rand() < q0:
                max_prob_index = np.argmax(transition_prob)
                next_index = next_index_meet_constrains[max_prob_index]
            else:
                # use roulette algorithm
                next_index = MultipleAntColonySystem.stochastic_accept(next_index_meet_constrains, transition_prob)

            # update pheromone matrix
            ant.graph.local_update_pheromone(ant.current_index, next_index)
            ant.move_to_next_index(next_index)

        # if you have gone through all the points, you need to go back to the depot
        if ant.index_to_visit_empty():
            ant.graph.local_update_pheromone(ant.current_index, 0)
            ant.move_to_next_index(0)

        # insert unvisited points to ensure that the path is feasible
        ant.insert(stop_event)

        # ant.index_to_visit_empty()==True means feasible
        if local_search is True and ant.index_to_visit_empty():
            ant.local_search_procedure(stop_event)

    @staticmethod
    def acs_time(new_graph, vehicle_num, ants_num, q0, beta, global_path_queue, path_found_queue, stop_event):
        # You can use vehicle_num vehicles at most, that is, the path contains the most vehicle_num + 1
        # depots to find the shortest path, vehicle_num is set to be consistent with the current best_path
        print('[acs_time]: start, vehicle_num %d' % vehicle_num)
        # initialize the pheromone matrix
        global_best_path = None
        global_best_distance = None
        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        while True:
            print('[acs_time]: new iteration')

            if stop_event.is_set():
                print('[acs_time]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, True,
                                          np.zeros(new_graph.node_num), q0, beta, stop_event)
                ants_thread.append(thread)
                ants.append(ant)

            # Here you can use the result method and wait for the thread to finish
            for thread in ants_thread:
                thread.result()

            ant_best_travel_distance = None
            ant_best_path = None
            # Determine whether the path found by the ant is feasible and better than the global path
            for ant in ants:

                if stop_event.is_set():
                    print('[acs_time]: receive stop event')
                    return

                # Get the current best path
                if not global_path_queue.empty():
                    info = global_path_queue.get()
                    while not global_path_queue.empty():
                        info = global_path_queue.get()
                    print('[acs_time]: receive global path info')
                    global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

                # The shortest path calculated by the ant
                if ant.index_to_visit_empty() and (
                        ant_best_travel_distance is None or ant.total_travel_distance < ant_best_travel_distance):
                    ant_best_travel_distance = ant.total_travel_distance
                    ant_best_path = ant.travel_path

            # Perform global update of pheromone here
            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            # Send the calculated current best path to macs
            if ant_best_travel_distance is not None and ant_best_travel_distance < global_best_distance:
                print('[acs_time]: ants\' local search found a improved feasible path, send path info to macs')
                path_found_queue.put(Path(ant_best_path, ant_best_travel_distance))

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    @staticmethod
    def acs_vehicle(new_graph, vehicle_num, ants_num, q0, beta, global_path_queue, path_found_queue, stop_event):
        # vehicle_num is set to one less than the current best_path
        print('[acs_vehicle]: start, vehicle_num %d' % vehicle_num)
        global_best_path = None
        global_best_distance = None

        # Initialize path and distance using nearest_neighbor_heuristic algorithm
        current_path, current_path_distance, _ = new_graph.nearest_neighbor(max_vehicle_num=vehicle_num)

        # Find the unvisited nodes in the current path
        current_index_to_visit = list(range(new_graph.node_num))
        for ind in set(current_path):
            current_index_to_visit.remove(ind)

        ants_pool = ThreadPoolExecutor(ants_num)
        ants_thread = []
        ants = []
        IN = np.zeros(new_graph.node_num)
        while True:
            print('[acs_vehicle]: new iteration')

            if stop_event.is_set():
                print('[acs_vehicle]: receive stop event')
                return

            for k in range(ants_num):
                ant = Ant(new_graph, 0)
                thread = ants_pool.submit(MultipleAntColonySystem.new_active_ant, ant, vehicle_num, False, IN, q0,
                                          beta, stop_event)

                ants_thread.append(thread)
                ants.append(ant)

            # Here you can use the result method and wait for the thread to finish
            for thread in ants_thread:
                thread.result()

            for ant in ants:

                if stop_event.is_set():
                    print('[acs_vehicle]: receive stop event')
                    return

                IN[ant.index_to_visit] = IN[ant.index_to_visit] + 1

                # The path found by the ant is compared with the current_path, can you use the vehicle_num
                # vehicles to access more nodes
                if len(ant.index_to_visit) < len(current_index_to_visit):
                    current_path = copy.deepcopy(ant.travel_path)
                    current_index_to_visit = copy.deepcopy(ant.index_to_visit)
                    current_path_distance = ant.total_travel_distance
                    # And set IN to 0
                    IN = np.zeros(new_graph.node_num)

                    # If this path is easy, it will be sent to macs_vrptw
                    if ant.index_to_visit_empty():
                        print('[acs_vehicle]: found a feasible path, send path info to macs')
                        path_found_queue.put(Path(ant.travel_path, ant.total_travel_distance))

            # Update the pheromone in new_graph, global
            new_graph.global_update_pheromone(current_path, current_path_distance)

            if not global_path_queue.empty():
                info = global_path_queue.get()
                while not global_path_queue.empty():
                    info = global_path_queue.get()
                print('[acs_vehicle]: receive global path info')
                global_best_path, global_best_distance, global_used_vehicle_num = info.get_path_info()

            new_graph.global_update_pheromone(global_best_path, global_best_distance)

            ants_thread.clear()
            for ant in ants:
                ant.clear()
                del ant
            ants.clear()

    def run_multiple_ant_colony_system(self, file_to_write_path=None):
        path_queue_for_figure = MPQueue()
        multiple_ant_colony_system_thread = Process(target=self._multiple_ant_colony_system,
                                                    args=(path_queue_for_figure, file_to_write_path,))
        multiple_ant_colony_system_thread.start()
        multiple_ant_colony_system_thread.join()

    def _multiple_ant_colony_system(self, path_queue_for_figure, file_to_write_path=None):
        start_time_total = time.time()

        # two queues time_what_to_do, vehicle_what_to_do tell the two threads acs_time,
        # acs_vehicle, what is the current best path, or let them stop calculating
        global_path_to_acs_time = Queue()
        global_path_to_acs_vehicle = Queue()

        # Another queue, path_found_queue, is a feasible path calculated by acs_time and acs_vehicle
        # that is better than best path
        path_found_queue = Queue()

        # Initialize using the nearest neighbor algorithm
        self.best_path, self.best_path_distance, self.best_vehicle_num = self.graph.nearest_neighbor()
        path_queue_for_figure.put(Path(self.best_path, self.best_path_distance))

        while True:
            print('[multiple_ant_colony_system]: new iteration')
            start_time_found_improved_solution = time.time()

            # the current best path is placed in the queue to inform acs_time and acs_vehicle of the current best_path
            global_path_to_acs_vehicle.put(Path(self.best_path, self.best_path_distance))
            global_path_to_acs_time.put(Path(self.best_path, self.best_path_distance))

            stop_event = Event()

            # acs_vehicle, try to explore with self.best_vehicle_num-1 vehicles, visit more nodes
            graph_for_acs_vehicle = self.graph.copy(self.graph.init_pheromone_val)
            acs_vehicle_thread = Thread(target=MultipleAntColonySystem.acs_vehicle,
                                        args=(graph_for_acs_vehicle, self.best_vehicle_num - 1, self.ants_num, self.q0,
                                              self.beta, global_path_to_acs_vehicle, path_found_queue, stop_event))

            # acs_time try to explore with self.best_vehicle_num vehicles and find a shorter path
            graph_for_acs_time = self.graph.copy(self.graph.init_pheromone_val)
            acs_time_thread = Thread(target=MultipleAntColonySystem.acs_time,
                                     args=(graph_for_acs_time, self.best_vehicle_num, self.ants_num, self.q0, self.beta,
                                           global_path_to_acs_time, path_found_queue, stop_event))

            # start acs_vehicle_thread and acs_time_thread, when they find a path that is easy and better than
            # the best path, it will be sent to macs
            print('[macs]: start acs_vehicle and acs_time')
            acs_vehicle_thread.start()
            acs_time_thread.start()

            best_vehicle_num = self.best_vehicle_num

            while acs_vehicle_thread.is_alive() and acs_time_thread.is_alive():

                # if no better results are found within the time, exit the program
                given_time = 5
                if time.time() - start_time_found_improved_solution > 60 * given_time:
                    stop_event.set()
                    print('*' * 50)
                    print('time is up: cannot find a better solution in given time(%d minutes)' % given_time)
                    print('it takes %0.3f second from multiple_ant_colony_system running' % (
                                time.time() - start_time_total))
                    print('the best path have found is:')
                    print(self.best_path)
                    print('best path distance is %f, best vehicle_num is %d' % (
                    self.best_path_distance, self.best_vehicle_num))
                    print('*' * 50)

                    return

                if path_found_queue.empty():
                    continue

                path_info = path_found_queue.get()
                print('[macs]: receive found path info')
                found_path, found_path_distance, found_path_used_vehicle_num = path_info.get_path_info()
                while not path_found_queue.empty():
                    path, distance, vehicle_num = path_found_queue.get().get_path_info()

                    if distance < found_path_distance:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                    if vehicle_num < found_path_used_vehicle_num:
                        found_path, found_path_distance, found_path_used_vehicle_num = path, distance, vehicle_num

                # if the distance of the found path is shorter, the current best path is updated
                if found_path_distance < self.best_path_distance:
                    # search for better results, update start_time
                    start_time_found_improved_solution = time.time()

                    print('*' * 50)
                    print('[macs]: distance of found path (%f) better than best path\'s (%f)'
                          % (found_path_distance, self.best_path_distance))
                    print('it takes %0.3f second from multiple_ant_colony_system running'
                          % (time.time() - start_time_total))
                    print('*' * 50)

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    # Notify the two threads acs_vehicle and acs_time, the currently found best_path
                    # and best_path_distance
                    global_path_to_acs_vehicle.put(Path(self.best_path, self.best_path_distance))
                    global_path_to_acs_time.put(Path(self.best_path, self.best_path_distance))

                # If there are fewer vehicles found by the two threads,
                # stop the two threads and start the next iteration
                # Send stop information to acs_time and acs_vehicle
                if found_path_used_vehicle_num < best_vehicle_num:
                    # search for better results, update start_time
                    start_time_found_improved_solution = time.time()
                    print('*' * 50)
                    print('[macs]: vehicle num of found path (%d) better than best path\'s (%d), found path distance '
                          'is %f' % (found_path_used_vehicle_num, best_vehicle_num, found_path_distance))
                    print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))
                    print('*' * 50)

                    self.best_path = found_path
                    self.best_vehicle_num = found_path_used_vehicle_num
                    self.best_path_distance = found_path_distance

                    print_graph(self.graph, self.best_path, self.best_path_distance, self.best_vehicle_num)

                    # stop threads acs_time and acs_vehicle
                    print('[macs]: send stop info to acs_time and acs_vehicle')
                    # notify threads acs_vehicle and acs_time, the currently found best_path and best_path_distance
                    stop_event.set()


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
