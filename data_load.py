import re
import numpy as np


class Node:
    def __init__(self, id: int, x: float, y: float, demand: float, ready_time: float, due_time: float,
                 service_time: float):
        super()
        self.id = id
        if id == 0:
            self.is_depot = True
        else:
            self.is_depot = False
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_time = due_time
        self.service_time = service_time


def get_data(fileName):
    f = open(fileName, "r")
    content = f.read()
    optimal_value = re.search("Optimal value: (\d+)", content, re.MULTILINE)
    if optimal_value is not None:
        optimal_value = optimal_value.group(1)
    else:
        optimal_value = re.search("Best value: (\d+)", content, re.MULTILINE)
        if optimal_value is not None:
            optimal_value = optimal_value.group(1)
    capacity = re.search("^CAPACITY : (\d+)$", content, re.MULTILINE).group(1)
    graph = re.findall(r"^(\d+) (\d+) (\d+)$", content, re.MULTILINE)
    demand = re.findall(r"^(\d+) (\d+)$", content, re.MULTILINE)
    graph = {int(a): (int(b), int(c)) for a, b, c in graph}
    demand = {int(a): int(b) for a, b in demand}
    capacity = int(capacity)
    optimal_value = int(optimal_value)
    return capacity, graph, demand, optimal_value


def load_data(file_path):
    node_list = []
    with open(file_path, 'rt') as f:
        count = 1
        for line in f:
            if count == 5:
                vehicle_num, vehicle_capacity = line.split()
                vehicle_num = int(vehicle_num)
                vehicle_capacity = int(vehicle_capacity)
            elif count >= 10:
                node_list.append(line.split())
            count += 1
    node_num = len(node_list)
    nodes = list(Node(int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]),
                      float(item[6])) for item in node_list)

    # 创建距离矩阵
    node_dist_mat = np.zeros((node_num, node_num))
    for i in range(node_num):
        node_a = nodes[i]
        # node_dist_mat[i][i] = 1e-8
        node_dist_mat[i][i] = 0
        for j in range(i + 1, node_num):
            node_b = nodes[j]
            node_dist_mat[i][j] = calculate_dist(node_a, node_b)
            node_dist_mat[j][i] = node_dist_mat[i][j]

    return node_num, nodes, node_dist_mat, vehicle_num, vehicle_capacity


def calculate_dist(node_a, node_b):
    return np.linalg.norm((node_a.x - node_b.x, node_a.y - node_b.y))
