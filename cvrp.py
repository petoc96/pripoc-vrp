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
file_name = "E-n22-k4.txt"
iterations = 1000
ants = 22


def generate_graph():
    # read graph and constrains from file
    capacity_limit, graph, demand, optimal_value = data_load.get_data(file_name)
    vertices = list(graph.keys())
    vertices.remove(1)
    points = list(graph.values())

    # get every possible edge
    edges = {(min(a, b), max(a, b)): np.sqrt((graph[a][0] - graph[b][0]) ** 2 + (graph[a][1] - graph[b][1]) ** 2) for
             a in graph.keys() for b in graph.keys()}
    # initialize feromones on each edge, initial value is 1
    feromones = {(min(a, b), max(a, b)): 1 for a in graph.keys() for b in graph.keys() if a != b}

    return points, vertices, edges, capacity_limit, demand, feromones, optimal_value


def solution_of_one_ant(vertices, edges, capacity_limit, demand, feromones):
    solution = list()
    # while there are cities left to visit
    while len(vertices) != 0:
        path = list()
        # start in random location
        city = np.random.choice(vertices)
        # update capacity/ reload
        capacity = capacity_limit - demand[city]
        # update path
        path.append(city)
        # remove already visited city from path
        vertices.remove(city)
        # while there are cities left to visit
        while len(vertices) != 0:
            # calculate probability for the remaining cities, according "that equation"
            # for each verticle apply the following function
            # (feromones of the edge (current city, another city) ^ lambda) --> trail level/ posteriori desirability
            # * (1/distance) ^ beta --> atractivenes of route/ prioi desirability
            probabilities = list(map(lambda x: ((feromones[(min(x, city), max(x, city))]) ** alfa) * (
                        (1 / edges[(min(x, city), max(x, city))]) ** beta), vertices))
            # normalize the probabilities
            probabilities = probabilities / np.sum(probabilities)
            # chose the next city from the non-uniform random distribution
            city = np.random.choice(vertices, p=probabilities)
            # update capacity
            capacity = capacity - demand[city]
            # check if we are still in the capacity limit
            if capacity > 0:
                path.append(city)
                vertices.remove(city)
            else:
                # return to depo
                break
        # append path and repeat
        solution.append(path)
    return solution

# calculates the lenght of the route
def rate_solution(solution, edges):
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


def update_feromone(feromones, solutions, best_solution):
    # get the average lenght of solutions
    Lavg = reduce(lambda x, y: x + y, (i[1] for i in solutions)) / len(solutions)
    # update each paths feromone: ro + Q
    feromones = {k: (ro + th / Lavg) * v for (k, v) in feromones.items()}
    solutions.sort(key=lambda x: x[1])
    if best_solution is not None:
        if solutions[0][1] < best_solution[1]:
            best_solution = solutions[0]
        for path in best_solution[0]:
            for i in range(len(path) - 1):
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = sigm / best_solution[1] + feromones[
                    (min(path[i], path[i + 1]), max(path[i], path[i + 1]))]
    else:
        best_solution = solutions[0]
    for l in range(sigm):
        paths = solutions[l][0]
        L = solutions[l][1]
        for path in paths:
            for i in range(len(path) - 1):
                feromones[(min(path[i], path[i + 1]), max(path[i], path[i + 1]))] = (sigm - (l + 1) / L ** (l + 1)) + \
                                                                                    feromones[(
                                                                                    min(path[i], path[i + 1]),
                                                                                    max(path[i], path[i + 1]))]
    return best_solution


def main():
    best_solution = None
    points, vertices, edges, capacity_limit, demand, feromones, optimal_value = generate_graph()

    for i in range(iterations):
        solutions = list()
        for _ in range(ants):
            solution = solution_of_one_ant(vertices.copy(), edges, capacity_limit, demand, feromones)
            solutions.append((solution, rate_solution(solution, edges)))
        best_solution = update_feromone(feromones, solutions, best_solution)
        print(str(i) + ":\t" + str(int(best_solution[1])) + "\t" + str(optimal_value))
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

    plt.scatter(x, y)
    for i in optimal_solution[0]:
        i = np.array(i) - 1
        # i = i-1
        x1 = np.concatenate(([x[0]], x[i], [x[0]]))
        y1 = np.concatenate(([y[0]], y[i], [y[0]]))
        plt.plot(x1, y1)
    plt.show()

    plt.scatter(x, y)
    for i in solution[0]:
        i = np.array(i) - 1
        # i = i-1
        x1 = np.concatenate(([x[0]], x[i], [x[0]]))
        y1 = np.concatenate(([y[0]], y[i], [y[0]]))
        plt.plot(x1, y1)
    plt.show()
