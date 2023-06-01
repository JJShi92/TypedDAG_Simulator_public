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


# Generates sets of DAG tasks

# Two types of processors, i.e., type A (M_A) and type B (M_B).
# For each type the number of available processors M_A/M_B \in {4, 8, 16}.
# For each set of tasks, the total utilization U_s \in [0, M] with step 5%xM, where M=M_A + M_B
# DRS package for allocating utilization for tasks, and nodes.
# Utilization for each task, the utilization U_{\tau} \in (0, U_s),
# in order to be possible to generate task with relatively high utilization.
# Number of nodes $N$ for each task is randomly selected \in [0.5 x (M_A +M_B), 5 x M_max], e.g., [4, 80]
# Additional 2 are added as common general starting node and ending node with 0 utilization.
# G(N, p) method is used to constructed the structure inside a DAG task,
# $p$ is the possibility that two nodes have precedence constraints \in [0.1, 0.9] or it can be set as [lb, lb+0.4]
# Period $T$ for each task \in [1, 2, 5, 10, 20, 50, 100, 200, 1000]
# Number of tasks for each set can have three modes, i.e., {[0.25 x M, M], [M, 2 x M], [0.5 x M_max, 2 x M_max]},
# where M_max = max(M_A, M_B).

# For federated scheduling or grouped constraints,
# The affinity of each vertex has to be defined.
# For each vertex the affinity can be a single processor or a cluster contains a set of processors.

class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.tsk_id = int # task id
        self.graph = defaultdict(list)  # edges
        self.weights = List[int]  # execution times
        self.priority = int # priority of the task
        self.priorities = List[int]  # priorities of each vertex
        self.deadlines = List[int]  # deadlines of each vertex
        self.utilization = float  # utilization of the task
        self.utilizations = List[float]  # utilizations of all vertices
        self.predecessors = defaultdict(list) # predecessors of each vertex
        self.period = int  # period of the task
        self.deadline = int  # deadline of the task
        self.affinity = defaultdict(list)  # affinity of each vertex
        self.cp = int  # length of the critical path

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def __getitem__(self, item):
        return self.graph.__getitem__(item)

    def __setitem__(self, key, value):
        self.graph.__setitem__(key, value)


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


# Find all these predecessors of all vertices
def find_predecessors(graph_org: Dict[int, List[int]], num_vertices):
    graph = copy.deepcopy(graph_org)
    predecessors_temp = defaultdict(list)

    for i in range(num_vertices):
        for j in range(len(graph[i])):
            predecessors_temp[graph[i][j]].append(i)

    # initialize the common start and end nod
    predecessors_temp[0]
    predecessors_temp[num_vertices-1]

    predecessors = copy.deepcopy(dict(sorted(predecessors_temp.items())))

    return predecessors



# Calculate the deadline for each vertex by considering the precedence constraints backforwardly
def calculate_deadlines(graph_org: Dict[int, List[int]], weights_org: List[int], deadline_org: int):
    graph = copy.deepcopy(graph_org)
    weights = copy.deepcopy(weights_org)
    deadline = copy.deepcopy(deadline_org)

    num_nodes = len(weights)
    deadlines = [deadline] * num_nodes
    for i in range(num_nodes - 2, -1, -1):
        successor_deadlines = [deadlines[j]-weights[j] for j in graph[i]]
        deadlines[i] = min(successor_deadlines)
    return deadlines


# msets: number of sets
# processor_a: number of processor A
# processor_b: number of processor B
# pc_prob_l: the lowest probability of two vertices have edge
# pc_prob_h: the highest probability of two vertices have edge
# The real probability \in [pc_prob_l, pc_prob_h]
# utilization: total utilization for a set of tasks
# sparse: the number of tasks for a set
# scale: the scale to keep all the parameters are integers
def generate_tsk_dict(msets, ntasks, nnodes, processor_a, processor_b, pc_prob, utilization, sparse, scale, preempt_times, main_mem_time):
    tasksets = []
    periods_all = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    num_tasks = 10
    for i in range(msets):
        taskset = []
        # generate the number of tasks
        if ntasks < 1:
            if sparse == 0:
                num_tasks = random.randint(0.5 * max(processor_a, processor_b), 2 * max(processor_a, processor_b))
            if sparse == 1:
                num_tasks = random.randint((processor_a + processor_b), 2 * (processor_a + processor_b))
            if sparse == 2:
                num_tasks = random.randint(0.25 * (processor_a + processor_b), (processor_a + processor_b))
        else:
            num_tasks = num_tasks

        # generate the period of each task
        periods = []
        # generate the number of nodes of each task
        num_nodes_all = []
        # generate the lower bound of the wcet of each task set
        util_lower_bound_set = []
        for tsk in range(num_tasks):
            period_temp = periods_all[random.randint(0, 8)]
            periods.append(period_temp)
            num_nodes_temp = random.randint(nnodes[0], nnodes[1])
            num_nodes_all.append(num_nodes_temp)
            # worst case data accessing time = main_mem_time * (preempt_times + 1)
            lower_bound_tsk_temp = num_nodes_temp * main_mem_time * (preempt_times + 1) / period_temp / scale
            util_lower_bound_set.append(lower_bound_tsk_temp)

        if sum(util_lower_bound_set) >= utilization:
            print("Too many preempt times, unable to generate the task set")
            return

        # drs(number, sum, upper_bound, lowe_bound)
        util_tasks = drs(num_tasks, utilization, None, util_lower_bound_set)
        j = 0
        while j < num_tasks:
            # add one common source node and one common end node
            # num_nodes = random.randint(0.5 * (processor_a + processor_b), 5 * max(processor_a, processor_b)) + 2
            num_nodes = num_nodes_all[j]
            g_temp = Graph(num_nodes)
            period = periods[j]
            g_temp.period = period * scale
            # implicit deadline
            g_temp.deadline = period * scale

            # generate the lower bound of the node of each task
            util_lower_bound_tsk = []
            for n in range(num_nodes):
                util_lower_bound_tsk.append(main_mem_time * (preempt_times + 1) / period / scale)

            # common source node and one common end node have 0 utilization
            util_nodes = drs(num_nodes-2, util_tasks[j], None, util_lower_bound_tsk)
            util_nodes.insert(0, 0)
            util_nodes.append(0)

            g_temp.utilization = util_tasks[j]
            g_temp.utilizations = util_nodes

            WCETs_float = [nd_u * period * scale for nd_u in util_nodes]
            WCETs_int = list(map(int, WCETs_float))

            # print(WCETs_int)
            g_temp.weights = WCETs_int
            g_temp.priorities = [0] * num_nodes

            # G(n, q) method to generate the precedence constraints
            pc_q = random.uniform(pc_prob[0], pc_prob[1])
            # define the graph structure
            struct = defaultdict(list)

            for source in range(1, num_nodes - 1):
                for dest in range(source+1, num_nodes):
                    if random.uniform(0, 1) < pc_q:
                        struct[source].append(dest)

            # check the connection to common source node
            all_list = range(1, num_nodes - 1)
            set_all_list = set(all_list)
            for node in range(1, num_nodes - 1):
                set_all_list = set_all_list - set(struct[node])

            # print(set_all_list)
            not_connected_source = list(set_all_list)
            if len(not_connected_source) > 0:
                for node in range(len(not_connected_source)):
                    struct[0].append(not_connected_source[node])

            # check the connection to common end node
            for node in range(1, num_nodes - 1):
                if len(struct[node]) == 0:
                    struct[node].append(num_nodes-1)
                if len(struct[node]) < 1:
                    print('Length3 : ', len(struct[node]))
                    print('something wrong 0!!!')

            # append the last item
            struct[num_nodes-1]

            g_temp.graph = copy.deepcopy(dict(sorted(struct.items())))

            for node in range(1, num_nodes - 1):
                if len(g_temp.graph[node]) < 1:
                    print('something wrong!!!')

            g_temp.predecessors = find_predecessors(g_temp.graph, num_nodes)

            # check if the longest path is longer than the period already
            # the total execution time is no longer than the period
            if sum(g_temp.weights) <= g_temp.period:
                g_temp.deadlines = calculate_deadlines(g_temp.graph, g_temp.weights, g_temp.deadline)
                g_temp.cp = int(find_longest_path(g_temp.graph, g_temp.weights)[1])
                g_temp.tsk_id = int(j)
                taskset.append(g_temp)
                j = j + 1
            else:
                # print('critical path has to be checked!')
                longest_path = find_longest_path(g_temp.graph, g_temp.weights)[1]
                if longest_path <= g_temp.period:
                    g_temp.cp = int(longest_path)
                    g_temp.deadlines = calculate_deadlines(g_temp.graph, g_temp.weights, g_temp.deadline)
                    g_temp.tsk_id = int(j)
                    taskset.append(g_temp)
                    j = j + 1

        # print("graph: ", taskset[0].graph)
        # print("weights: ", taskset[0].weights)
        # print("predecessors: ", taskset[0].predecessors)

        # set the priority for each task
        # start from 1, the lower number the higher priority
        # Rate montonic approach, same period -> task with higher utilization has higher priority (lower number)
        taskset_temp = copy.deepcopy(taskset)
        sorted_index = [i[0] for i in sorted(enumerate(taskset_temp), key=lambda x: (x[1].period, -x[1].utilization))]
        for tsk in range(num_tasks):
            taskset[sorted_index[tsk]].priority = int(tsk + 1)

        # append each set to a utilization specified big set
        tasksets.append(taskset)

    return tasksets

# generate_tsk_dict(1, 4, [50, 100], 4, 16, [0.2, 0.4], 10, 0, 10**6, 100)

