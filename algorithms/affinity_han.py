# The approach proposed by Meiling Han
# with different processor assignment method
import copy
from drs import drs
import numpy as np
import random
from typing import *
from operator import itemgetter
from collections import defaultdict
import math
from collections import deque
import time
import sys
sys.path.append('../')
from algorithms import misc

# Calculate the longest path of a give DAG task
def find_longest_path(graph: Dict[int, List[int]], weights: List[float]):
    base_cost = [weights[v] for v in graph.keys()]
    def dfs(graph, vertex: int, cost: List[float], visited: List[bool]):
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


# HGSH method to schedule a heavy task on $pro_a$ type A processors and $pro_b$ processors
# Return the feasibility: 1-schedulable (response time <= period/deadline) 0-otherwise
def hgsh(task_org, type_org, processors_org):
    task = copy.deepcopy(task_org)
    type_info = copy.deepcopy(type_org)

    # store the processor information
    # index 0: the number of processor A
    # index 1: the number of processor B
    processors = copy.deepcopy(processors_org)

    # only type B processors
    if processors[0] == 0:
        vol_sum = type_info.utilizationB/processors[1]*task.period
    # only type A processors
    elif processors[1] == 0:
        vol_sum = type_info.utilizationA/processors[0]*task.period
    # both type of processors
    else:
        vol_sum = (type_info.utilizationA/processors[0] + type_info.utilizationB/processors[1]) * task.period

    # calculate the scaled longest path
    scaled_weights = copy.deepcopy(task.weights)
    for i in range(type_info.V):
        if scaled_weights[i] > 0:
            scaled_weights[i] = scaled_weights[i] * (1 - (1/(processors[int(type_info.typed[i]) - 1])))

    response_time = vol_sum + find_longest_path(task.graph, scaled_weights)[1]

    return response_time


def emu_assign(task_org, type_org, available_a, available_b, penalty_a, penalty_b):
    task = copy.deepcopy(task_org)
    type_info = copy.deepcopy(type_org)

    # impossible to be scheduled
    if type_info.utilizationA > available_a or type_info.utilizationB > available_b:
        return False

    ub_a = int(available_a + 1)
    ub_b = int(available_b + 1)

    current_best = [0, 0]
    penalty = 3

    for i in range(math.ceil(type_info.utilizationA), ub_a):
        for j in range(math.ceil(type_info.utilizationB), ub_b):
            penalty_temp = i * penalty_a + j * penalty_b
            if (hgsh(task, type_info, [i, j]) <= task.period) and (penalty_temp < penalty):
                penalty = penalty_temp
                current_best[0] = i
                current_best[1] = j
                break
    #FIXME: if sum(current_best) != 0:
    if current_best[0] != 0 and current_best[1] != 0:
        return current_best
    else:
        return False


def greedy_assign(task_org, type_org, available_a, available_b, penalty_a, penalty_b):
    task = copy.deepcopy(task_org)
    type_info = copy.deepcopy(type_org)

    current_a = math.ceil(type_info.utilizationA)
    current_b = math.ceil(type_info.utilizationB)

    # impossible to be scheduled
    if current_a > available_a or current_b > available_b:
        return False, [available_a - current_a, available_b - current_b]

    response_time = hgsh(task, type_info, [current_a, current_b])
    while response_time > task.period and current_a <= available_a and current_b <= available_b:
        temp_rt_a = hgsh(task, type_info, [(current_a + 1), current_b])
        temp_rt_b = hgsh(task, type_info, [current_a, (current_b + 1)])

        if (response_time - temp_rt_a - penalty_a) > (response_time - temp_rt_b - penalty_b):
            response_time = temp_rt_a
            current_a = current_a + 1
        else:
            response_time = temp_rt_b
            current_b = current_b + 1

    if response_time < task.period:
        return True, [current_a, current_b]
    else:
        return False, [-1, -1]


# the sum of ceiling for task with higher priorities: heavy_a
# task_hp = [[task, typed, R_i]...]
def sum_hp_a(time_t, tasks_hp):
    sum = 0
    for i in range(len(tasks_hp)):
        sum = sum + math.ceil((((time_t + tasks_hp[i][1]) / tasks_hp[i][0][0].period) - tasks_hp[i][0][1].utilizationB)) * tasks_hp[i][0][1].utilizationB * tasks_hp[i][0][0].period
    return sum


# the sum of ceiling for task with higher priorities: heavy_b
# task_hp = [[task, typed, R_i]...]
def sum_hp_b(time_t, tasks_hp):
    sum = 0
    for i in range(len(tasks_hp)):
        sum = sum + math.ceil((((time_t + tasks_hp[i][1]) / tasks_hp[i][0][0].period) - tasks_hp[i][0][1].utilizationA)) * tasks_hp[i][0][1].utilizationA * tasks_hp[i][0][0].period
    return sum


# schedule a light task on one a core and one b core
def sched_light_fix(light_task, task_hp_a, task_hp_b):
    wcet_new = sum(light_task[0].weights)
    # if both a and b processor is empty
    if len(task_hp_a) == 0 and len(task_hp_b) == 0:
        return wcet_new

    time_t = 1
    response_time = 0
    start_time = time.time()
    # while time_t <= light_task.period and time.time() - start_time < 1200:
    # it requires both A and B core
    if light_task[1].utilizationA > 0 and light_task[1].utilizationB > 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_a(time_t, task_hp_b) + sum_hp_b(time_t, task_hp_a)

            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    # it only requires A core
    elif light_task[1].utilizationA > 0 and light_task[1].utilizationB == 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_b(time_t, task_hp_a)
            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    # it only requires B core
    elif light_task[1].utilizationA == 0 and light_task[1].utilizationB > 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_a(time_t, task_hp_b)

            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    return False


# schedule light task on A/B core along with other tasks
# light_ab = [task, type_info]
def sched_share_light(light_ab, partitioned_ab_org):
    partitioned_ab = copy.deepcopy(partitioned_ab_org)

    for i in range(len(partitioned_ab[0])):
        for j in range(len(partitioned_ab[1])):
            response = sched_light_fix(light_ab, partitioned_ab[0][i], partitioned_ab[1][j])
            if response:
                new_partition_a = []
                new_partition_a.append(light_ab)
                new_partition_a.append(response)
                new_partition_b = []
                new_partition_b.append(light_ab)
                new_partition_b.append(response)

                if light_ab[1].utilizationA > 0 and light_ab[1].utilizationB > 0:
                    partitioned_ab[0][i].append(new_partition_a)
                    partitioned_ab[1][j].append(new_partition_b)
                    return True, partitioned_ab, [i, j]
                elif light_ab[1].utilizationA > 0 and light_ab[1].utilizationB == 0:
                    partitioned_ab[0][i].append(new_partition_a)
                    return True, partitioned_ab, [i, -1]
                elif light_ab[1].utilizationA == 0 and light_ab[1].utilizationB > 0:
                    partitioned_ab[1][j].append(new_partition_b)
                    return True, partitioned_ab, [-1, i]

    return False, partitioned_ab, [-1, -1]


# calculate the number of processor A if only processor A is required
# task_org = [task, type_info]
def only_processor_a(task_org, available_a):
    task = copy.deepcopy(task_org)
    current_available_a = copy.deepcopy(available_a)
    for i in range(1, current_available_a+1):
        if hgsh(task[0], task[1], [i, 0]) <= task[0].period:
            return int(i)
    return 0


# calculate the number of processor B if only processor B is required
# task_org = [task, type_info]
def only_processor_b(task_org, available_b):
    task = copy.deepcopy(task_org)
    current_available_b = copy.deepcopy(available_b)
    for i in range(1, current_available_b+1):
        if hgsh(task[0], task[1], [0, i]) <= task[0].period:
            return int(i)
    return 0


# mod 0: emu assignment
# mod 1: greedy method
def sched_han(taskset_org, typed_org, available_a, available_b, mod):
    taskset = copy.deepcopy(taskset_org)
    typed_info = copy.deepcopy(typed_org)
    current_available_a = copy.deepcopy(available_a)
    current_available_b = copy.deepcopy(available_b)

    # Record the current index for both processor A and B under allocation
    # current_index = [current_a_index, current_b_index]
    current_index = []
    current_a_index = 0
    current_b_index = copy.deepcopy(available_a)
    current_index.append(current_a_index)
    current_index.append(current_b_index)

    # Affinities for tasks
    affinities = defaultdict(list)

    penalty_a = 1 / current_available_a
    penalty_b = 1 / current_available_b

    light_tasks = []

    for i in range(0, len(taskset)):
        # check if the task only require one type of processor
        # only require processor A:
        if typed_info[i].utilizationB == 0 and typed_info[i].utilizationA > 1:
            used_a = only_processor_a([taskset[i], typed_info[i]], current_available_a)
            if used_a > 0:
                affinities[i], current_index = misc.assign_affinity_task(current_index, [used_a, 0])
                current_available_a = current_available_a - used_a
            else:
                return False, affinities, [-1, 0]

        # only require processor B:
        elif typed_info[i].utilizationA == 0 and typed_info[i].utilizationB > 1:
            used_b = only_processor_b([taskset[i], typed_info[i]], current_available_b)
            if used_b > 0:
                affinities[i], current_index = misc.assign_affinity_task(current_index, [0, used_b])
                current_available_b = current_available_b - used_b
            else:
                return False, affinities, [0, -1]

        # try to divide the tasks into heavy and light
        # heavy task: density > 1 -> volume > period
        elif taskset[i].utilization > 1:

            # different processor assignment methods
            if mod == 1:
                assigned = emu_assign(taskset[i], typed_info[i], current_available_a, current_available_b, penalty_a, penalty_b)
                # update the current available processors
                if assigned:
                    affinities[i], current_index = misc.assign_affinity_task(current_index, assigned)

                    current_available_a = current_available_a - assigned[0]
                    current_available_b = current_available_b - assigned[1]
                else:
                    return False, affinities, [-1, -1]

            else:
                assigned = greedy_assign(taskset[i], typed_info[i], current_available_a, current_available_b, penalty_a, penalty_b)
                # update the current available processors
                if assigned[0]:
                    affinities[i], current_index = misc.assign_affinity_task(current_index, assigned[1])

                    current_available_a = current_available_a - assigned[1][0]
                    current_available_b = current_available_b - assigned[1][1]
                else:
                    return False, affinities, [current_available_a - assigned[1][0], current_available_b - assigned[1][1]]

            # no sufficient processors for heavy tasks
            if current_available_a < 0 or current_available_b < 0:
                return False, affinities, [current_available_a, current_available_b]

        # store all light tasks here
        else:
            light_tasks.append([taskset[i], typed_info[i]])

    # if there is only one light task, check it directly.
    if len(light_tasks) == 0:
        return True, affinities, current_index

    if len(light_tasks) == 1:
        if light_tasks[0][1].utilizationA > 0 and light_tasks[0][1].utilizationB == 0 and current_available_a > 0:
            # schedulable by default
            affinities[light_tasks[0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [1, 0])
            return True, affinities, current_index
        elif light_tasks[0][1].utilizationB > 0 and light_tasks[0][1].utilizationA == 0 and current_available_b > 0:
            # schedulable by default
            affinities[light_tasks[0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [0, 1])
            return True, affinities, current_index
        elif light_tasks[0][1].utilizationB > 0 and light_tasks[0][1].utilizationA > 0 and current_available_a > 0 and current_available_b > 0:
            # schedulable by default
            affinities[light_tasks[0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [1, 1])
            return True, affinities, current_index
        else:
            return False, affinities, [current_available_a - 1, current_available_b - 1]

    else:
        # sort light tasks
        light_tasks.sort(key=lambda x: x[0].priority)
        # schedula light tasks using our method
        partition_a = []
        partition_b = []
        for a in range(current_available_a):
            partition_a.append(deque())
        for b in range(current_available_b):
            partition_b.append(deque())
        partition_ab = []
        partition_ab.append(partition_a)
        partition_ab.append(partition_b)

        shared_start_index = copy.deepcopy(current_index)

        for l in range(len(light_tasks)):
            new_partitioned_ab = sched_share_light(light_tasks[l], partition_ab)
            if new_partitioned_ab[0]:
                partition_ab = new_partitioned_ab[1]
                used_shared_index = [x + y for x, y in zip(shared_start_index, new_partitioned_ab[2])]
                if light_tasks[l][1].utilizationA == 0:
                    used_shared_index[0] = -1
                if light_tasks[l][1].utilizationB == 0:
                    used_shared_index[1] = -1
                affinities[light_tasks[l][0].tsk_id], current_index = misc.assign_affinity_light_task(current_index, used_shared_index)
            else:
                return False, affinities, new_partitioned_ab[2]

    return True, affinities, current_index
