import re


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
