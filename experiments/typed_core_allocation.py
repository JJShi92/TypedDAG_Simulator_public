import copy
from drs import drs
import numpy as np
import random
from typing import *
from operator import itemgetter
from collections import defaultdict
import time

# seed_value = 123
# random.seed(seed_value)


# Allocate vertices to two types of processors
# Two types of processors, i.e., type A (M_A) and type B (M_B).
# For each type the number of available processors M_A/M_B \in {4, 8, 16}.
class Type:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.typed = defaultdict(int) # the type of processors allocation, 1: type A; 2: type B; 0: undefined
        self.utilizationA = float # the utilization of all vertices in type A cores
        self.utilizationB = float # the utilization of all vertices in type B cores
        self.cpA = int  # critical path of all vertices in type A cores
        self.cpB = int  # critical path of all vertices in type B cores


# Calculate the longest path of a give DAG task
def find_longest_path(graph: Dict[int, List[int]], weights: List[int]):
    base_cost = [weights[v] for v in graph.keys()]
    def dfs(graph, vertex: int, cost: List[int], visited: List[bool]):
        visited[vertex] = True
        for connected_vertex in graph[vertex]:
            if not visited[connected_vertex]:
                dfs(graph, connected_vertex, cost, visited)
            if cost[vertex] > base_cost[vertex]+cost[connected_vertex]:
                cost[vertex] = cost[vertex]
            else:
                cost[vertex] = base_cost[vertex]+cost[connected_vertex]

    cost = copy.deepcopy(base_cost)
    visited = [False for v in graph.keys()]
    for vertex in graph.keys():
        if not visited[vertex]:
            dfs(graph, vertex, cost, visited)

    return max(enumerate(cost), key=lambda item: item[1])


# Generate the type for each node by following different mode.
# mod_1: P_\ell controls the skewness of the skewed tasks
# e.g., 10% nodes for the task is assigned on A core, and others on B core (heavy^b task)
# mod_2: the percentage of heavy^a or heavy^b tasks, e.g., 0%, 25%, 50%, 75%, and 100%
# mod_3: if allow a task only require one type of processor:
# i.e., 0: not allowed; 1: allowed, the percentage can be defined by mod_2 (if needed)
def generate_tsk_type(msets, task_sets_org, processor_a, processor_b, mod_1, mod_2, mod_3):
    type_sets = []

    task_sets = copy.deepcopy(task_sets_org)

    threshods = [0.1, 0.05, 0.01]
    percentages = [1, 0.75, 0.5, 0.25, 0]

    threshod_temp = threshods[mod_1]
    percent = percentages[mod_2]

    for i in range(msets):
        type_set = []
        for j in range(len(task_sets[i])):
            num_nodes = task_sets[i][j].V
            type_temp = Type(num_nodes)
            single_node = random.randint(1, num_nodes-1)
            # normal situation, each task require both types of processors
            if mod_3 == 0:
                # heavy^a or heavy^b task
                if random.uniform(0, 1) < percent:
                    # decide if it is heavy^a or heavy^b task
                    if random.uniform(0, 1) < 0.5:
                        # heavy^b
                        threshold = threshod_temp
                    else:
                        threshold = 1 - threshod_temp
                else:
                    # the threshold follows the M_a/(M_a + M_b)
                    threshold = processor_a / (processor_a + processor_b)

            # only allow heavy A when #A >> #B
            elif mod_3 == 1:
                if random.uniform(0, 1) < percent:
                    threshold = 1 - threshod_temp
                else:
                    # the threshold follows the M_a/(M_a + M_b)
                    threshold = processor_a / (processor_a + processor_b)
            elif mod_3 == 2:
                # the special case that each task only has one node that is allocate to processor B when #A >> #B
                # or randomly A or B when #A == #B
                single_node = random.randint(1, num_nodes-1)
            # since we assume # processor A > # processor B
            # we only allow task can only require processor A here
            else:
                # Only processor A
                if random.uniform(0, 1) < percent:
                    # all the nodes are assigned on processor A
                    threshold = 1
                else:
                    # follows the threshold percentage
                    threshold = 1 - threshod_temp

            # temp weights for vertices only in processor A
            temp_weights_a = copy.deepcopy(task_sets[i][j].weights)
            # temp weights for vertices only in processor B
            temp_weights_b = copy.deepcopy(task_sets[i][j].weights)

            for w in range(len(task_sets[i][j].weights)):
                if temp_weights_a[w] != temp_weights_b[w]:
                    print("something wrong")

            # define the type for each node
            if mod_3 == 2:
                if processor_a > processor_b:
                    # only heavy a
                    for node in range(0, num_nodes):
                        if node != single_node:
                            type_temp.typed[node] = 1
                            temp_weights_b[node] = 0

                        else:
                            type_temp.typed[node] = 2
                            temp_weights_a[node] = 0
                else:
                    if random.uniform(0, 1) < 0.5:
                        # heavy a
                        for node in range(0, num_nodes):
                            if node != single_node:
                                type_temp.typed[node] = 1
                                temp_weights_b[node] = 0

                            else:
                                type_temp.typed[node] = 2
                                temp_weights_a[node] = 0
                    else:
                        # heavy b
                        for node in range(0, num_nodes):
                            if node == single_node:
                                type_temp.typed[node] = 1
                                temp_weights_b[node] = 0

                            else:
                                type_temp.typed[node] = 2
                                temp_weights_a[node] = 0
            # unbalanced mode
            elif threshold <= 0.1:
                for node in range(0, num_nodes):
                    if random.uniform(0, 1) < threshold or node == single_node:
                        type_temp.typed[node] = 1
                        temp_weights_b[node] = 0

                    else:
                        type_temp.typed[node] = 2
                        temp_weights_a[node] = 0
            elif threshold >= 0.9:
                for node in range(0, num_nodes):
                    if random.uniform(0, 1) < threshold and node!= single_node:
                        type_temp.typed[node] = 1
                        temp_weights_b[node] = 0

                    else:
                        type_temp.typed[node] = 2
                        temp_weights_a[node] = 0
            else:
                for node in range(0, num_nodes):
                    if random.uniform(0, 1) < threshold:
                        type_temp.typed[node] = 1
                        temp_weights_b[node] = 0

                    else:
                        type_temp.typed[node] = 2
                        temp_weights_a[node] = 0

            # In the following some additional information is calculated for future usage
            # calculate the L_i^a and L_i^b
            La = find_longest_path(task_sets[i][j].graph, temp_weights_a)[1]
            Lb = find_longest_path(task_sets[i][j].graph, temp_weights_b)[1]

            if La == task_sets[i][j].period/3 or La == task_sets[i][j].period/2:
                La += 1
            if Lb == task_sets[i][j].period/3 or Lb == task_sets[i][j].period/2:
                Lb += 1

            type_temp.cpA = La
            type_temp.cpB = Lb

            # Accumulative execution time of vertices on type A and B
            type_temp.utilizationA = sum(temp_weights_a) / task_sets[i][j].period
            type_temp.utilizationB = sum(temp_weights_b) / task_sets[i][j].period

            # Check the common source and end node
            # if they are allocated to the wrong typed of processor
            if type_temp.utilizationA == 0:
                # no A core is required
                type_temp.typed[0] = 2
                type_temp.typed[num_nodes-1] = 2

            if type_temp.utilizationB == 0:
                # no B core is required
                type_temp.typed[0] = 1
                type_temp.typed[num_nodes-1] = 1

            type_set.append(type_temp)

        type_sets.append(type_set)

    return type_sets
