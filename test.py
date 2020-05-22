from model.graph import Graph
from aco import ACO


if __name__ == '__main__':
    file_path = 'data/c101.txt'
    ants_num = 100
    max_iter = 1000
    beta = 2
    q0 = 0.1

    graph = Graph(file_path)
    aco = ACO(graph, ants_num=ants_num, max_iter=max_iter, beta=beta, q0=q0)

    aco.run()
