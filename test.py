from model.graph import Graph
from aco import ACO
from multiple_aco import MultipleAntColonySystem


if __name__ == '__main__':
    file_path = 'data/c101.txt'
    ants_num = 10
    max_iter = 1000
    beta = 2
    q0 = 0.1

    graph = Graph(file_path)
    m_aco = MultipleAntColonySystem(graph, ants_num=ants_num, beta=beta, q0=q0)
    m_aco.run_multiple_ant_colony_system()

