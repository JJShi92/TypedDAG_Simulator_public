# Some functions that can help the calculations of the simulator
from collections import defaultdict
import math
import copy
from drs import drs
import numpy as np
import random
from typing import *
from operator import itemgetter
import time

# Helper function to calculate the LCM of two numbers
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


# Function to find the LCM of a list of numbers
def find_lcm(numbers):
    lcm_result = numbers[0]
    for i in range(1, len(numbers)):
        lcm_result = lcm(lcm_result, numbers[i])
    return lcm_result


# Assign affinity for single task
# current_index = [current_a_index, current_b_index]
# required_processors = [requires_a_processors, required_b_processors]
def assign_affinity_task(current_index_org, required_processors):
    current_index = copy.deepcopy(current_index_org)

    affinity = defaultdict(list)

    processor_a_list = list(range(current_index[0], current_index[0]+required_processors[0]))
    processor_b_list = list(range(current_index[1], current_index[1]+required_processors[1]))

    affinity[1]
    affinity[2]

    if required_processors[0] > 0:
        affinity[1] = processor_a_list

    if required_processors[1] > 0:
        affinity[2] = processor_b_list

    current_index = [x + y for x, y in zip(current_index, required_processors)]

    return affinity, current_index


# Assign affinity for single task
# current_index = [current_a_index, current_b_index]
# required_processors = [requires_a_processors, required_b_processors]
def assign_affinity_single_heavy_task(current_index_org, required_processors):
    current_index = copy.deepcopy(current_index_org)

    affinity = defaultdict(list)

    if required_processors[0] >= 0:
        processor_a_list = list(range(required_processors[0], required_processors[0] + 1))
        affinity[1] = processor_a_list

    elif required_processors[1] >= 0:
        processor_b_list = list(range(required_processors[1], required_processors[1] + 1))
        affinity[2] = processor_b_list

    if required_processors[0] > current_index[0]:
        current_index[0] = required_processors[0]

    if required_processors[1] > current_index[1]:
        current_index[1] = required_processors[1]

    return affinity, current_index


# Assign affinity for single light task
# current_index = [current_a_index, current_b_index]
# required_processors = [requires_a_processors, required_b_processors]
def assign_affinity_light_task(current_index_org, required_processors):
    current_index = copy.deepcopy(current_index_org)

    affinity = defaultdict(list)

    if required_processors[0] >= 0:
        processor_a_list = list(range(required_processors[0], required_processors[0] + 1))
    else:
        processor_a_list = []
    if required_processors[1] >= 0:
        processor_b_list = list(range(required_processors[1], required_processors[1] + 1))
    else:
        processor_b_list = []

    affinity[1] = processor_a_list
    affinity[2] = processor_b_list

    if required_processors[0] > current_index[0]:
        current_index[0] = required_processors[0]

    if required_processors[1] > current_index[1]:
        current_index[1] = required_processors[1]

    return affinity, current_index


# Assign affinity for single task
# current_index = [current_a_index, current_b_index]
# required_processors = [requires_a_processors, required_b_processors]
# mod-0: heavy_a shared b core, mod-1: heavy_b shared a core
def assign_affinity_mix_heavy_task(current_index_org, required_processors, mod):
    current_index = copy.deepcopy(current_index_org)

    affinity = defaultdict(list)

    if mod == 0:
        processor_a_list = list(range(current_index[0], current_index[0] + required_processors[0]))
        affinity[1] = processor_a_list
        processor_b_list = list(range(required_processors[1], required_processors[1] + 1))
        affinity[2] = processor_b_list
        current_index[0] += required_processors[0]


    if mod == 1:
        processor_a_list = list(range(required_processors[0], required_processors[0] + 1))
        affinity[1] = processor_a_list
        processor_b_list = list(range(current_index[1], current_index[1] + required_processors[1]))
        affinity[2] = processor_b_list

        current_index[1] += required_processors[1]

    return affinity, current_index


# Find the maximum index of used a and b cores
def find_max_ab_index(affinities):
    max_value_key1 = float('-inf')
    max_value_key2 = float('-inf')

    for inner_dict in affinities.values():
        if len(inner_dict[1]) > 0:
            max_value_key1 = max(max_value_key1, max(inner_dict[1]))
        if len(inner_dict[2]) > 0:
            max_value_key2 = max(max_value_key2, max(inner_dict[2]))

    return max_value_key1, max_value_key2


def list_all_ab_index(affinities):
    set1 = set(affinities[0][1])
    set2 = set(affinities[0][2])

    for inner_dict in affinities.values():
        set1 = set1 | set(inner_dict[1])
        set2 = set2 | set(inner_dict[2])

    return set1, set2


# Due to the unused A cores, the index for all these used B cores have to be re-organized
def adjust_unused_cores(affinities_org, unused_a_cores):
    affinities = copy.deepcopy(affinities_org)
    for i in range(len(affinities)):
        affinities[i][2] = [x - unused_a_cores for x in affinities[i][2]]

    return affinities


# Calculate the utilization on each core
def cores_utilizations(affinities, typed_org, total_processors):
    cores_util = defaultdict()
    for i in range(total_processors):
        cores_util[i] = 0

    for i in range(len(affinities)):
        if len(affinities[i][1]) == 1:
            cores_util[affinities[i][1][0]] += typed_org[i].utilizationA
        if len(affinities[i][2]) == 1:
            cores_util[affinities[i][2][0]] += typed_org[i].utilizationB

    for i in range(total_processors):
        if cores_util[i] > 1:
            print("The affinity is incorrect! ", i, cores_util[i])
        else:
            print("core utilization: ", i, cores_util[i])


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


# Convert the task set with the average case execution time
def average_case_convertor_taskset(task_set_org, avg_ratio):
    taskset = copy.deepcopy(task_set_org)
    for i in range(len(taskset)):
        taskset[i].weights = [int(weight * avg_ratio) for weight in taskset[i].weights]
        taskset[i].utilization = taskset[i].utilization * avg_ratio
        taskset[i].utilizations = [ut * avg_ratio for ut in taskset[i].utilizations]
        taskset[i].deadlines = calculate_deadlines(taskset[i].graph, taskset[i].weights, taskset[i].deadline)
        taskset[i].cp = taskset[i].cp * avg_ratio

    return taskset


# Convert the typed information with the average case execution time
def average_case_convertor_typed(typed_info_org, avg_ratio):
    typed_info = copy.deepcopy(typed_info_org)

    for i in range(len(typed_info)):
        typed_info[i].utilizationA = typed_info[i].utilizationA * avg_ratio
        typed_info[i].utilizationB = typed_info[i].utilizationB * avg_ratio

        typed_info[i].cpA = typed_info[i].cpA * avg_ratio
        typed_info[i].cpB = typed_info[i].cpB * avg_ratio

    return typed_info