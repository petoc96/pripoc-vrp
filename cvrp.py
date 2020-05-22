import math
import data_load
import random
import numpy as np
from functools import reduce
import sys
import getopt
import matplotlib.pyplot as plt

alfa = 2
beta = 5
sigm = 3
ro = 0.8
th = 80
q0 = 0.1
#file_name = "data/E-n22-k4.txt"
file_name = "data/c101.txt"
iterations = 100
ants = 22
#ants = 100


def generate_graph():
    # read graph and constrains from file
    node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity = data_load.load_data(file_name)

    edges = {}
    # get every possible edge
    for i in range(1, node_num):
        for j in range(i, node_num):
            edges[(min(i, j), max(i, j))] = node_dist_mat[i][j]

    # initialize feromones on each edge, initial value is 1
    feromones = {(min(a, b), max(a, b)): 1 for a in range(node_num) for b in range(node_num) if a != b}
    demand = {a.id: a.demand for a in nodes}
    capacity_limit = vehicle_capacity
    vertices = list(range(1, node_num))
    points = [(a.x, a.y) for a in nodes]
    times = [(a.ready_time, a.due_time) for a in nodes]

    return points, vertices, edges, capacity_limit, demand, feromones, times


def solution_of_one_ant(vertices, edges, capacity_limit, demand, feromones, times):
    solution = list()
    # while there are cities left to visit
    while len(vertices) != 0:
        path = list()
        # start in random location
        city = np.random.choice(vertices)
        # update capacity / reload
        capacity = capacity_limit - demand[city]
        # update path
        path.append(city)
        # remove already visited city from path
        vertices.remove(city)
### we should reset timer here
        time = 0
        tries = 0
        # while there are cities left to visit
        while len(vertices) != 0:
#### so (1/distance) must be changed to solve TW, something like (target delivery time - ( current time + travel time)) -> should be preferably less
            # calculate probability for the remaining cities, according "that equation"
            # for each verticle apply the following function
            # (feromones of the edge (current city, another city) ^ lambda) --> trail level/ posteriori desirability
            # * (1/distance) ^ beta --> atractivenes of route/ prioi desirability
            probabilities = list(map(lambda x: ((feromones[(min(x, city), max(x, city))]) ** alfa) * (
                    (1 / edges[(min(x, city), max(x, city))]) ** beta), vertices))
# 1/(distance + waiting time )^s1 * 1/(max arrival - min arrival)^s2 
#                    (1 / ( edges[(min(x, city), max(x, city))] + max(times[x][0] - edges[(min(x, city), max(x, city))] - time, 0))**4 * 1/ (times[x][1] - times[x][0])**2 ) ** beta), vertices))

            # normalize the probabilities
            #probabilities = list(map(lambda x: max(x,0),probabilities))
            probabilities = probabilities / np.sum(probabilities)

            prevcity = city

 
            city = get_next_city(vertices, probabilities,city)
            if not check_conditions(capacity,demand,city,edges,prevcity,time,times):
                 city = get_next_city(vertices, probabilities,city)
                 if not check_conditions(capacity,demand,city,edges,prevcity,time,times):
                     city = get_next_city(vertices, probabilities,city)
                     if not check_conditions(capacity,demand,city,edges,prevcity,time,times):
                         #return to depo
                         break
            else:
                 path.append(city)
                 vertices.remove(city)
                 wait_time = max(times[city][0] - edges[(min(prevcity, city), max(prevcity, city))] - time, 0)
                 time = time + edges[(min(prevcity, city), max(prevcity, city))] + wait_time

        # append path and repeat
        solution.append(path)
    return solution

def check_conditions(capacity,demand,city,edges,prevcity,time,times):
    # update capacity
    capacity = capacity - demand[city]
    # check if we are still in the capacity limit
### also target delivery time should be checked here
    if capacity > 0 and (time + edges[(min(prevcity, city), max(prevcity, city))] ) < times[city][1]:
        return True
    else:
        return False

def get_next_city(vertices, probabilities,current):
    if np.random.rand() < q0:
        max_prob_index = np.argmax(probabilities)
        city = vertices[max_prob_index]
    else:
        # calculate N and max fitness value
        N = len(vertices)

        # normalize
        sum_prob = np.sum(probabilities)
        norm_prob = probabilities/sum_prob

        # select: O(1)
        while True:
            # randomly select an individual with uniform probability
            ind = int(N * random.random())
            if random.random() <= norm_prob[ind]:
               city = vertices[ind]
               break
    return city
#            # chose the next city from the non-uniform random distribution
#            city = np.random.choice(vertices, p=probabilities)

# calculates the lenght of the route
def rate_solution(solution, edges):
#######################################this must be updated to solve TW too, expect if we still only optimize for distance
    s = 0
    for i in solution:
        # start from depo
        a = 1
        for j in i:
            b = j
            s = s + edges[(min(a, b), max(a, b))]
            a = b
        # return to depo
        b = 1
        s = s + edges[(min(a, b), max(a, b))]
    return s


# Elitist update rule-> the best solution deposits pheromone on its trail
def update_feromone(feromones, solutions, best_solution):
    # get the average lenght of solutions
    Lavg = reduce(lambda x, y: x + y, (i[1] for i in solutions)) / len(solutions)
    # evaporation update each paths feromone: (1const + 2const/(avg lenght))*pheromone -> the less is the avg lenght
    # the less is the evaporation ratio
    feromones = {k: (ro + th / Lavg) * v for (k, v) in feromones.items()}
    # elitist update
    solutions.sort(key=lambda x: x[1])
    if best_solution is not None:
        if solutions[0][1] < best_solution[1]:
            best_solution = solutions[0]
        for path in best_solution[0]:
            for i in range(len(path) - 1):
                # update the best path += sigma/weight
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = sigm / best_solution[1] + feromones[
                    (min(path[i], path[i + 1]), max(path[i], path[i + 1]))]
    else:
        best_solution = solutions[0]
    for l in range(sigm):
        paths = solutions[l][0]
        L = solutions[l][1]
        for path in paths:
            for i in range(len(path) - 1):
                # for the first pheromone update, update the sigma best paths, by weighting its orders
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = (sigm - (l + 1) / L ** (l + 1)) + \
                                                                                    feromones[(
                                                                                        min(path[i], path[i + 1]),
                                                                                        max(path[i], path[i + 1]))]
    return best_solution


def main():
    best_solution = None
    points, vertices, edges, capacity_limit, demand, feromones, times = generate_graph()

    for i in range(iterations):
        solutions = list()
        for _ in range(ants):
            solution = solution_of_one_ant(vertices.copy(), edges, capacity_limit, demand, feromones, times)
            solutions.append((solution, rate_solution(solution, edges)))
        best_solution = update_feromone(feromones, solutions, best_solution)
        print(str(i) + ":\t" + str(int(best_solution[1])) + "\t")
    return points, best_solution


if __name__ == "__main__":
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "f:a:b:s:r:t:i:n:", ["fileName=",
                                                              "alpha=", "beta=", "sigma=", "rho=", "theta=",
                                                              "iterations=", "numberOfAnts="])
    except getopt.GetoptError:
        print("""use: python ACO_CVRP.py 
            -f <fileName> 
            -a <alpha> 
            -b <beta> 
            -s <sigma> 
            -r <rho> 
            -t <theta>
            -i <iterations>
            -n <numberOfAnts>
            Default values:
            fileName: E-n22-k4.txt
            alpha: 80
            beta: 5
            sigma: 3
            rho: 0.8
            theta: 80
            iterations: 1000
            number of ants: 22""")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-a", "--alpha"):
            alfa = float(arg)
        elif opt in ("-b", "--beta"):
            beta = float(arg)
        elif opt in ("-s", "--sigma"):
            sigm = float(arg)
        elif opt in ("-r", "--rho"):
            ro = float(arg)
        elif opt in ("-t", "--theta"):
            th = float(arg)
        elif opt in ("-f", "--fileName", "--file"):
            file_name = str(arg)
        elif opt in ("-i", "--iterations"):
            iterations = int(arg)
        elif opt in ("-n", "--numberOfAnts"):
            ants = int(arg)

    print("file name:\t" + str(file_name) +
          "\nalpha:\t" + str(alfa) +
          "\nbeta:\t" + str(beta) +
          "\nsigma:\t" + str(sigm) +
          "\nrho:\t" + str(ro) +
          "\ntheta:\t" + str(th) +
          "\niterations:\t" + str(iterations) +
          "\nnumber of ants:\t" + str(ants))

    points, solution = main()

    print("Solution: " + str(solution))
    if file_name == "E-n22-k4.txt":
        optimal_solution = ([[18, 21, 19, 16, 13], [17, 20, 22, 15], [14, 12, 5, 4, 9, 11], [10, 8, 6, 3, 2, 7]], 375)
        print("Optimal solution: " + str(optimal_solution))

    x = np.array([i[0] for i in points])
    y = np.array([i[1] for i in points])

    # plt.scatter(x, y)
    # for i in optimal_solution[0]:
    #     i = np.array(i) - 1
    #     # i = i-1
    #     x1 = np.concatenate(([x[0]], x[i], [x[0]]))
    #     y1 = np.concatenate(([y[0]], y[i], [y[0]]))
    #     plt.plot(x1, y1)
    # plt.show()

    plt.scatter(x, y)
    for i in solution[0]:
        i = np.array(i)
        # i = i-1
        x1 = np.concatenate(([x[0]], x[i], [x[0]]))
        y1 = np.concatenate(([y[0]], y[i], [y[0]]))
        plt.plot(x1, y1)
    plt.show()
