import numpy as np
import math
import copy
import random
from collections import deque
from collections import OrderedDict
import sys
from typing import *
from operator import itemgetter
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.append('../')
from generators import generator_pure_dict
from algorithms import memory as mem
from algorithms import affinity_raw as raw
from algorithms import affinity_han as han
from algorithms import affinity_improved_fed as imp_fed
from algorithms import misc
from generators import data_requests
from generators import typed_core_allocation


# Each task has multiple vertices
# task_id
# vertex_id
# job_start_time: identify the starting time of the whole job (not the sub-job of the vertex)
# priority: for Fixed Priority scheduling algorithm. When RM is applied, the priority is the same as the period
# requested_data: a list to store all the requested data addresses
# pure_execution_time: except the wcet of data accesses
# start_time: of the sub-job of the vertex
# rest_execution_time: the rest execution time to finish the sub-job of the vertex
# deadline: of the sub-job of the vertex
# successors
# affinity
# preempt_times: the remaining times that the sub-job of the vertex can be preempted
class Vertex:
    def __init__(self, task_id, vertex_id, job_start_time, priority, requested_data, pure_execution_time, start_time, rest_execution_time, deadline, successors, affinity, preempt_times):
        self.task_id = task_id
        self.vertex_id = vertex_id
        self.job_start_time = job_start_time
        self.priority = priority
        self.requested_data = requested_data
        self.pure_execution_time = pure_execution_time
        self.start_time = start_time
        self.rest_execution_time = rest_execution_time
        self.deadline = deadline
        self.successors = successors
        self.affinity = affinity
        self.preempt_times = preempt_times


# Different processor may point to a same ready queue
# The current scheduling task contains all the information of the vertex
# The busy time start records the starting time of the current vertex
# The schedule record contains the busy time slots and the corresponding task id and vertex id
class Processor:
    def __init__(self, ready_queue_id):
        self.ready_queue_id = ready_queue_id
        self.current_scheduling_vertex = Vertex
        self.busy_time_start = int
        self.schedule_record = list()


# Find the non-repeated cluster
# Example input affinities = {'0': [1, 2], '1': [1, 2], '2': [3], '3': [3], '4': [1, 2]}
# Output [[1, 2], [3]]
def find_cluster(one_task_affinities):
    affinities = copy.deepcopy(one_task_affinities)
    unique_clusters = set()
    for k, v in affinities.items():
        # Convert the list to a tuple
        tuple_v = tuple(v)
        # Check if the tuple has already been added to the set
        if tuple_v not in unique_clusters:
            # Add the tuple to the set
            unique_clusters.add(tuple_v)

    # Convert to list
    unique_lists = []
    for unique_tuple in unique_clusters:
        unique_list = list(unique_tuple)
        unique_lists.append(unique_list)

    return unique_lists


# The affinity for each vertex has been determined in a dictionary by different algorithms,
# e.g., {1: [0, 1]} determines that in this task all vertices in Type A cores can be executed on processor 0 and 1.
# The input affinities contain the affinities of all tasks in a set
# The affinity for each task contains N elements, N is the number of types of cores
# For each task, it can be executed on multiple A/B cores
# The corresponding ready queues have to be determined
# This function provides the core to ready queue mapping
# Due to the federated schedule, multiple cores can point to the same ready queue
def ready_queues_initialization(affinities_org, maximum_core):
    affinities = copy.deepcopy(affinities_org)
    ready_queues = []
    core_to_ready_queue = dict()
    unique_clusters = set()
    ready_queue_id = 0
    for i in range(len(affinities)):
        for k, v in affinities[i].items():
            # Convert the list to a tuple
            tuple_v = tuple(v)
            # Check if the tuple has already been added to the set
            if tuple_v not in unique_clusters:
                # Add the tuple to the set
                unique_clusters.add(tuple_v)
                # Create a new ready queue
                ready_queues.append(deque())
                # Create the core to ready queue mapping
                for core in range(len(tuple_v)):
                    if tuple_v[core] >= maximum_core:
                        print("Allocated core is out of the range!")
                        return None
                    else:
                        core_to_ready_queue[tuple_v[core]] = ready_queue_id
                ready_queue_id += 1

    return ready_queues, core_to_ready_queue


# Generate the real execution time
# first minus the effect of worst case data accessing time
# Based on normal distribution
# wcet: worst case execution time
# min: the minimal execution time (best case execution time)
# mean: average case execution time
# Usually, we set the mean to 0.5: average execution time = 0.5 x wcet
# std_dev: standard deviation
# Normally, we set std_dev < 0.5 to ensure the real execution times are not far away from the mean
def gen_real_ec(wcet_org, preempt_times, req_prob, main_mem_time, min, mean, std_dev):
    wcet = copy.deepcopy(wcet_org)
    random_prop = random.normalvariate(mean, std_dev)

    # bound constraints to set the lower and upper bound of the real execution time
    if random_prop < min:
        random_prop = min
    if random_prop > 1:
        random_prop = 1

    # count the number of data addresses that can be requested
    data_num = sum(1 for element in req_prob if element != 0)

    return int((wcet - (preempt_times + 1) * main_mem_time * data_num) * random_prop)


# Initialize the predecessors for all vertices of tasks
# mod0: initialize the predecessors for all the tasks in the set
# mod1: only initialize the predecessors of one task
def predecessors_initialization(taskset_org, task_id, mod):
    taskset = copy.deepcopy(taskset_org)
    predecessors_set = []
    predecessors_task = defaultdict(list)

    if mod == 0:
        for i in range(len(taskset)):
            predecessors_set.append(taskset[i].predecessors)
        return predecessors_set

    else:
        predecessors_task = taskset[task_id].predecessors
        return predecessors_task


# The time for accessing requested data:
def data_access_time(addresses, mem_hierarchy):
    access_time = 0
    for i in range(len(addresses)):
        # The common source and end node request no data
        if addresses[i] == hex(0):
            access_time += 0
        else:
            access_time += mem_hierarchy.get(addresses[i])

    return access_time, mem_hierarchy


# Abort the vertex that will miss its deadline anyway
def abort_vertex(tasks, n_tsk, predecessors, Ready_queues, core_to_queue_mapping, affinities, data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, mod):
    abt_tsk_id = n_tsk.task_id
    abt_vertex_id = n_tsk.vertex_id

    # Release the precedence constraints of its successors
    abt_successors = n_tsk.successors
    new_ready_vertices = []

    # the finished vertex is not the common end node
    if len(abt_successors) > 0:
        for f_v in range(len(abt_successors)):
            if abt_vertex_id in predecessors[abt_tsk_id][abt_successors[f_v]]:
                predecessors[abt_tsk_id][abt_successors[f_v]].remove(abt_vertex_id)
            else:
                print("Abort vertex, the predecessor constraint is wrong!")
            # If the successor has no remained constraint, it will be added into the corresponding ready queue
            if len(predecessors[abt_tsk_id][abt_successors[f_v]]) == 0:
                # the successor's vertex id
                s_vertex_id = abt_successors[f_v]
                new_ready_vertices.append(s_vertex_id)
                # the start time of the whole job to calculate the relative deadline
                j_start_time = n_tsk.job_start_time
                # obtain the ready queue affinity
                rq_affinity = core_to_queue_mapping[
                    affinities[abt_tsk_id][typed_info[abt_tsk_id].typed[s_vertex_id]][0]]
                # generate the real execution time
                if s_vertex_id == 0 or s_vertex_id == tasks[abt_tsk_id].V - 1:
                    rec = tasks[abt_tsk_id].weights[s_vertex_id]
                    if rec != 0:
                        print("Node execution time error, abort!")
                else:
                    # rec = tasks[abt_tsk_id].weights[s_vertex_id] - ((preempt_times + 1) * main_mem_time)
                    rec = gen_real_ec(tasks[abt_tsk_id].weights[s_vertex_id], preempt_times, data_requests[abt_tsk_id].req_prob[s_vertex_id], main_mem_time, min_ratio, avg_ratio, std_dev)
                    if rec < 0:
                        print("Abort execution time error!")
                        return
                requested_data_temp = gen_requested_data(data_requests[abt_tsk_id].req_data[s_vertex_id],
                                                         data_requests[abt_tsk_id].req_prob[s_vertex_id])
                sub_job = Vertex(abt_tsk_id, s_vertex_id, j_start_time, tasks[abt_tsk_id].priority,
                                 requested_data_temp, rec,
                                 current_time,
                                 rec,
                                 tasks[abt_tsk_id].deadlines[s_vertex_id] + j_start_time,
                                 tasks[abt_tsk_id].graph[s_vertex_id],
                                 rq_affinity, preempt_times)
                rec_all += rec
                # append to the corresponding ready queue
                Ready_queues[rq_affinity].append(sub_job)
                # Sort the corresponding ready queue according to their deadlines
                if mod == 0:
                    Ready_queues[rq_affinity] = deque(sorted(Ready_queues[rq_affinity], key=lambda x: x.deadline))
                else:
                    Ready_queues[rq_affinity] = deque(sorted(Ready_queues[rq_affinity], key=lambda x: (x.priority, x.deadline)))

    return predecessors, Ready_queues, rec_all, new_ready_vertices


# Update the all_checked status
def update_all_checked(affinities, all_checked):

    for i in range(len(affinities)):
        all_checked[affinities[i]] = True

    return all_checked


# Find all successors of a given vertex
# Include both direct successors and indirect successors
def find_all_successors(graph_org, vertex_org):
    graph = copy.deepcopy(graph_org)
    vertex = copy.deepcopy(vertex_org)

    successors = set()
    stack = [vertex]

    while stack:
        current_vertex = stack.pop()

        if current_vertex in successors:
            continue

        successors.add(current_vertex)
        neighbors = graph.get(current_vertex, [])

        for neighbor in neighbors:
            stack.append(neighbor)

    successors.remove(vertex)

    return list(sorted(successors))


# Define the requested data for a vertex when it is added to ready queue:
def gen_requested_data(possible_requested_data, requested_data_probability):
    requested_data = []
    for i in range(len(possible_requested_data)):
        if random.uniform(0, 1) < requested_data_probability[i]:
            requested_data.append(possible_requested_data[i])

    return requested_data


# The general Typed DAG schedule simulator with Fixed Priority scheduling algorithm (Rate-Motonic (RM))
# The processor allocations, i.e., affinities, are given by different federated scheduling algorithms
# Discard version: sub-jobs that can potentially miss their deadlines are discarded in advance,
# i.e., deleted by the readyqueue checker, and do not be scheduled.
def typed_dag_schedule_rm_discard_sim(tasks_org, affinities_org, typed_org, data_requests, total_processors, max_time, scale, preempt_times, memory_org, main_mem_time, avg_ratio, min_ratio, std_dev):
    tasks = copy.deepcopy(tasks_org)
    affinities = copy.deepcopy(affinities_org)
    typed_info = copy.deepcopy(typed_org)
    data_requests = copy.deepcopy(data_requests)
    memory_hierarchy = copy.deepcopy(memory_org)

    rec_all = 0

    # record all the possible deadline misses
    deadline_misses = []

    # Initialize the ready queues and find the core to ready queue mapping
    Ready_queues, core_to_queue_mapping = ready_queues_initialization(affinities, total_processors)

    # Initialize all the available processors
    processors = []
    for i in range(total_processors):
        processors.append(Processor(core_to_queue_mapping[i]))
        processors[i].current_scheduling_vertex = None

    # Check the next check time for each cluster
    current_time_temp = []
    # initialize the current time temp
    for pro in range(total_processors):
        current_time_temp.append(scale)

    # Initialize the predecessors
    predecessors = predecessors_initialization(tasks, 0, 0)

    # Release tasks' common source nodes according to their periods
    # Deadlines are updated according to the corresponding release time
    # Real duration can be shorter than the wcet
    for t in range(0, max_time, scale):

        # The initial release for all tasks' common source nodes
        if t == 0:
            predecessors = predecessors_initialization(tasks, 0, 0)
            for tsk in range(len(tasks)):
                for vtx in range(tasks[tsk].V):
                    # release all the vertices without any precedences
                    if len(predecessors[tsk][vtx]) == 0:
                        # obtain the ready queue affinity
                        rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                        # generate the real execution time
                        if vtx == 0 or vtx == tasks[tsk].V-1:
                            rec = tasks[tsk].weights[vtx]
                            if rec != 0:
                                print("Node execution time error v0! ")
                        else:
                            # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                            rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, data_requests[tsk].req_prob[vtx], main_mem_time, min_ratio, avg_ratio, std_dev)
                        requested_data_temp = gen_requested_data(data_requests[tsk].req_data[vtx], data_requests[tsk].req_prob[vtx])
                        sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, requested_data_temp, rec, t, rec, tasks[tsk].deadlines[vtx], tasks[tsk].graph[vtx], rq_affinity, preempt_times)

                        rec_all += rec
                        # append to the corresponding ready queue
                        Ready_queues[rq_affinity].append(sub_job)

        # Release tasks' common source nodes according to their periods
        else:
            for tsk in range(len(tasks)):
                if t % tasks[tsk].period == 0:
                    # A new job of the task is released
                    # All these sub-jobs from previous period have to be cleaned

                    # remove the current running sub-jobs
                    for pro in range(total_processors):
                        if processors[pro].current_scheduling_vertex is not None:
                            if processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > t:
                                deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                        processors[pro].current_scheduling_vertex.vertex_id,
                                                        t, pro])
                            elif processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time == t:
                                processors[pro].schedule_record[-1].append(t)

                    # clean the ready queue
                    for q in range(len(Ready_queues)):
                        if len(Ready_queues[q]) > 0:
                            for v_id in range(len(Ready_queues[q])):
                                if Ready_queues[q][v_id].task_id == tsk:
                                    deadline_misses.append([tsk, Ready_queues[q][v_id].vertex_id, t])
                            Ready_queues[q] = deque([vet for vet in Ready_queues[q] if vet.task_id != tsk])

                    # clean the predecessors
                    for p in range(len(predecessors[tsk])):
                        if len(predecessors[tsk][p]) > 0:
                            deadline_misses.append([tsk, p, t])

                    # initialize the predecessors of the corresponding task
                    predecessors[tsk] = predecessors_initialization(tasks, tsk, 1)
                    for vtx in range(tasks[tsk].V):
                        # release all the vertices without any precedences
                        if len(predecessors[tsk][vtx]) == 0:
                            # obtain the ready queue affinity
                            rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                            # generate the real execution time
                            if vtx == 0 or vtx == tasks[tsk].V - 1:
                                rec = tasks[tsk].weights[vtx]
                                if rec != 0:
                                    print("Node execution time error tp")
                            else:
                                # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                                rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, data_requests[tsk].req_prob[vtx], main_mem_time, min_ratio, avg_ratio, std_dev)
                            requested_data_temp = gen_requested_data(data_requests[tsk].req_data[vtx],
                                                                     data_requests[tsk].req_prob[vtx])
                            sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, requested_data_temp, rec, t,
                                             rec, tasks[tsk].deadlines[vtx] + t, tasks[tsk].graph[vtx],
                                             rq_affinity, preempt_times)
                            rec_all += rec
                            # append to the corresponding ready queue
                            Ready_queues[rq_affinity].append(sub_job)

        # Sort the corresponding ready queue according to their priorities
        for q in range(len(Ready_queues)):
            if len(Ready_queues[q]) > 0:
                Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: (x.priority, x.deadline)))

        current_time = t

        while(current_time < t + scale):
            # Check whether the scheduled vertex has finished its execution at the moment
            for pro in range(total_processors):

                if processors[pro].current_scheduling_vertex is not None:
                    # Deadline miss
                    if current_time > processors[pro].current_scheduling_vertex.deadline or (current_time == processors[pro].current_scheduling_vertex.deadline and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > current_time):
                        print("Deadline miss task is scheduled, please check!")
                        # Abort the current vertex
                        deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                processors[pro].current_scheduling_vertex.vertex_id, current_time, pro])
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)

                        # Still in the same period
                        if current_time < processors[pro].current_scheduling_vertex.job_start_time + tasks[processors[pro].current_scheduling_vertex.task_id].period:
                            abt_tsk = processors[pro].current_scheduling_vertex
                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk, predecessors, Ready_queues, core_to_queue_mapping, affinities,
                                         data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 1)
                        # All its successors have missed their deadlines
                        else:
                            # do not need to update its predecessor
                            abt_all_successors = find_all_successors(tasks[abt_tsk.task_id].graph, abt_tsk.vertex_id)
                            deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                    abt_all_successors, current_time,
                                                    pro])

                        processors[pro].current_scheduling_vertex = None

                    # The vertex has finished its execution
                    elif current_time == (processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time):
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)
                        # Update the precedence constraint
                        f_tsk_id = processors[pro].current_scheduling_vertex.task_id
                        f_vertex_id = processors[pro].current_scheduling_vertex.vertex_id
                        f_successors = processors[pro].current_scheduling_vertex.successors
                        # the finished vertex is not the common end node
                        if len(f_successors) != 0:
                            for f_v in range(len(f_successors)):
                                if f_vertex_id in predecessors[f_tsk_id][f_successors[f_v]]:
                                    predecessors[f_tsk_id][f_successors[f_v]].remove(f_vertex_id)
                                else:
                                    print("ERROR: The predecessor has been removed, Please check!", current_time, processors[pro].current_scheduling_vertex.job_start_time, tasks[f_tsk_id].weights[f_vertex_id], f_vertex_id, f_tsk_id, predecessors_initialization(tasks, f_tsk_id, 1)[f_successors[f_v]])
                                # If the successor has no remained constraint, it will be added into the corresponding ready queue
                                if len(predecessors[f_tsk_id][f_successors[f_v]]) == 0:
                                    # the successor's vertex id
                                    s_vertex_id = f_successors[f_v]
                                    # the start time of the whole job to calculate the relative deadline
                                    j_start_time = processors[pro].current_scheduling_vertex.job_start_time
                                    # obtain the ready queue affinity
                                    rq_affinity = core_to_queue_mapping[affinities[f_tsk_id][typed_info[f_tsk_id].typed[s_vertex_id]][0]]
                                    # generate the real execution time
                                    if s_vertex_id == 0 or s_vertex_id == tasks[f_tsk_id].V - 1:
                                        rec = tasks[f_tsk_id].weights[s_vertex_id]
                                        if rec != 0:
                                            print("Node execution time error ", rec, s_vertex_id, tasks[f_tsk_id].V)
                                    else:
                                        # rec = tasks[f_tsk_id].weights[s_vertex_id] - ((preempt_times + 1) * main_mem_time)
                                        if tasks[f_tsk_id].weights[s_vertex_id] < 0:
                                            print("Weight is wrong!")
                                        rec = gen_real_ec(tasks[f_tsk_id].weights[s_vertex_id], preempt_times, data_requests[f_tsk_id].req_prob[s_vertex_id], main_mem_time, min_ratio, avg_ratio, std_dev)
                                        if rec < 0:
                                            print("Execution time error! ", rec)
                                            return

                                    if rec > tasks[f_tsk_id].weights[s_vertex_id]:
                                        print("Execution time error rec")

                                    requested_data_temp = gen_requested_data(data_requests[f_tsk_id].req_data[s_vertex_id],
                                                                             data_requests[f_tsk_id].req_prob[s_vertex_id])
                                    sub_job = Vertex(f_tsk_id, s_vertex_id, j_start_time, tasks[f_tsk_id].priority, requested_data_temp, rec, current_time,
                                                 rec, tasks[f_tsk_id].deadlines[s_vertex_id] + j_start_time, tasks[f_tsk_id].graph[s_vertex_id],
                                                 rq_affinity, preempt_times)

                                    rec_all += rec
                                    # append to the corresponding ready queue
                                    Ready_queues[rq_affinity].append(sub_job)

                                    # Sort the corresponding ready queue according to their deadlines
                                    Ready_queues[rq_affinity] = deque(sorted(Ready_queues[rq_affinity], key=lambda x: (x.priority, x.deadline)))

                        # Set the current executing vertex in the processor as None
                        processors[pro].current_scheduling_vertex = None

            # List RM is applied, pickup the vertex from the ready queue with the highest priority (smallest period)
            # Either the previously executing vertex has been finished
            # Or the new released vertex can preempt the current executing vertex
            all_checked = [True] * len(Ready_queues)
            while any(all_checked):
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        if all_checked[q]:
                            all_checked[q] = False
                            single_queue_checked = False
                            while not single_queue_checked:
                                if len(Ready_queues[q]) <= 0:
                                    single_queue_checked = True
                                else:
                                    single_queue_checked = True
                                    sub_id = 0
                                    while sub_id < len(Ready_queues[q]):
                                        # it is impossible to be scheduled with the minimal data accessing time in the worst case
                                        # abort it in advance
                                        data_num_temp = len(Ready_queues[q][sub_id].requested_data)
                                        if data_num_temp * main_mem_time * (Ready_queues[q][sub_id].preempt_times + 1) + current_time + Ready_queues[q][sub_id].pure_execution_time > Ready_queues[q][sub_id].deadline:
                                            abt_tsk_rq = copy.deepcopy(Ready_queues[q][sub_id])
                                            del Ready_queues[q][sub_id]
                                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk_rq, predecessors, Ready_queues, core_to_queue_mapping, affinities, data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 1)

                                            if len(new_ready_vertices) > 0:
                                                abt_tsk_id = abt_tsk_rq.task_id
                                                for s_id in range(len(new_ready_vertices)):
                                                    # The processors with the Ready_queues[rq_affinity] have to be checked again
                                                    rq_affinity = core_to_queue_mapping[affinities[abt_tsk_id][typed_info[abt_tsk_id].typed[new_ready_vertices[s_id]]][0]]

                                                    all_checked[rq_affinity] = True
                                                    if rq_affinity == q:
                                                        single_queue_checked = False
                                            sub_id = 0
                                        else:
                                            sub_id += 1
                    else:
                        all_checked[q] = False

            # Sort the corresponding ready queue according to their deadlines
            for q in range(len(Ready_queues)):
                if len(Ready_queues[q]) > 0:
                    Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: (x.priority, x.deadline)))

            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex == None and len(Ready_queues[processors[pro].ready_queue_id]) > 0:
                    # select the new task
                    n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                    n_tsk.start_time = current_time
                    data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data, memory_hierarchy)
                    # The new vertex can potentially finish its execution
                    if current_time + n_tsk.pure_execution_time + data_accessing_time <= n_tsk.deadline:
                        n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)
                        processors[pro].current_scheduling_vertex = n_tsk
                        # Record the starting corresponding processor's busy time
                        if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                            print("missed the finish time 1")
                        busy_start_temp = []
                        busy_start_temp.append(n_tsk.task_id)
                        busy_start_temp.append(n_tsk.vertex_id)
                        busy_start_temp.append(n_tsk.deadline)
                        busy_start_temp.append(n_tsk.rest_execution_time)
                        busy_start_temp.append(current_time)

                        processors[pro].schedule_record.append(busy_start_temp)
                    # The new vertex will miss the deadline any way
                    else:
                        print("Ready queue check is wrong, the sub-job in the ready queue is not schedulable!", n_tsk.task_id, n_tsk.vertex_id, current_time, n_tsk.pure_execution_time, data_accessing_time, n_tsk.job_start_time, n_tsk.deadline)

                if len(Ready_queues[processors[pro].ready_queue_id]) > 0 and processors[pro].current_scheduling_vertex is not None:
                    # if preemptive:
                    preemptable = False
                    # Check if the currently executing vertex can be preempted due to the FP-RM
                    if processors[pro].current_scheduling_vertex.preempt_times > 0:
                        if Ready_queues[processors[pro].ready_queue_id][0].priority < processors[pro].current_scheduling_vertex.priority:
                            preemptable = True
                        if Ready_queues[processors[pro].ready_queue_id][0].priority == processors[pro].current_scheduling_vertex.priority and Ready_queues[processors[pro].ready_queue_id][0].deadline < processors[pro].current_scheduling_vertex.deadline:
                            preemptable = True

                    if preemptable:
                        # schedule the new vertex with the highest priority
                        n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                        # add the data accessing time
                        data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data, memory_hierarchy)
                        n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)

                        # The new vertex can potentially finish its execution
                        if current_time + n_tsk.rest_execution_time <= n_tsk.deadline:
                            # modify the preempted vertex
                            preempted_tsk = processors[pro].current_scheduling_vertex
                            # check if the vertex is preempted when accessing data
                            if current_time - preempted_tsk.start_time > preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time:
                                preempted_tsk.pure_execution_time = preempted_tsk.pure_execution_time - (current_time - preempted_tsk.start_time - (preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time))
                            if preempted_tsk.pure_execution_time < 0:
                                print("Not preemptable sub job is preempted! ")

                            # update the available preemptable times
                            preempted_tsk.preempt_times -= 1
                            # Record the end corresponding processor's busy time
                            processors[pro].schedule_record[-1].append(current_time)
                            # Append the preempted vertex back and sort the corresponding ready queue
                            Ready_queues[processors[pro].ready_queue_id].append(preempted_tsk)
                            Ready_queues[processors[pro].ready_queue_id] = deque(sorted(Ready_queues[processors[pro].ready_queue_id], key=lambda x: (x.priority, x.deadline)))

                            n_tsk.start_time = current_time
                            processors[pro].current_scheduling_vertex = n_tsk
                            # Record the starting corresponding processor's busy time
                            if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                                print("missed the finish time")
                            busy_start_temp = []
                            busy_start_temp.append(n_tsk.task_id)
                            busy_start_temp.append(n_tsk.vertex_id)
                            busy_start_temp.append(n_tsk.deadline)
                            busy_start_temp.append(n_tsk.rest_execution_time)
                            busy_start_temp.append(current_time)
                            processors[pro].schedule_record.append(busy_start_temp)
                        else:
                            print("Wrong job in the ready queue! Preempt the current executing sub-job!")

            # Update the current time for next check
            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex is not None:
                    # FIXME: Check the min(deadline, possible finish time) rather than only the possible finish time
                    # to make sure that the following vertices can potentially finish their execution
                    if processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time > processors[pro].current_scheduling_vertex.deadline:
                        print("Some thing wrong, wrong vertex is scheduled !", processors[pro].current_scheduling_vertex.task_id, processors[pro].current_scheduling_vertex.vertex_id, processors[pro].current_scheduling_vertex.deadline)
                    current_time_temp[pro] = min(processors[pro].current_scheduling_vertex.deadline, processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time)
                    # current_time_temp[pro] = processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time
                else:
                    current_time_temp[pro] = t + scale

            current_time = min(current_time_temp)

            # The next checking time is out of the range of maximum simulating time
            # Stop the simulation
            if current_time >= max_time:
                current_time = max_time
                for pro in range(total_processors):
                    if processors[pro].current_scheduling_vertex is not None:
                        processors[pro].schedule_record[-1].append(current_time)
                        if current_time > processors[pro].current_scheduling_vertex.deadline or current_time > processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time:
                            # deadline miss
                            deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                    processors[pro].current_scheduling_vertex.vertex_id, current_time,
                                                    pro])
                        elif current_time < processors[pro].current_scheduling_vertex.deadline:
                            print("Maximum simulation time is set incorrectly")

                # clean all the ready queues:
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        for v_id in range(len(Ready_queues[q])):
                            deadline_misses.append([Ready_queues[q][v_id].task_id, Ready_queues[q][v_id].vertex_id, t])
                    Ready_queues[q].clear()

    print("Average released utilization: ", rec_all/total_processors/max_time)

    return processors, deadline_misses



# The general Typed DAG schedule simulator with Fixed Priority scheduling algorithm (Rate-Motonic (RM))
# The processor allocations, i.e., affinities, are given by different federated scheduling algorithms
# Normal version: sub-jobs that can potentially miss their deadlines are not checked in advance,
# i.e., only when they are missing their deadlines, they will be aborted.
def typed_dag_schedule_rm_norm_sim(tasks_org, affinities_org, typed_org, data_requests, total_processors, max_time, scale, preempt_times, memory_org, main_mem_time, l1_cache_time, avg_ratio, min_ratio, std_dev):
    tasks = copy.deepcopy(tasks_org)
    affinities = copy.deepcopy(affinities_org)
    typed_info = copy.deepcopy(typed_org)
    data_requests = copy.deepcopy(data_requests)
    memory_hierarchy = copy.deepcopy(memory_org)

    rec_all = 0

    # record all the possible deadline misses
    deadline_misses = []

    # Initialize the ready queues and find the core to ready queue mapping
    Ready_queues, core_to_queue_mapping = ready_queues_initialization(affinities, total_processors)

    # Initialize all the available processors
    processors = []
    print(total_processors, len(core_to_queue_mapping))
    for i in range(total_processors):
        processors.append(Processor(core_to_queue_mapping[i]))
        processors[i].current_scheduling_vertex = None

    # Check the next check time for each cluster
    current_time_temp = []
    # initialize the current time temp
    for pro in range(total_processors):
        current_time_temp.append(scale)

    # Initialize the predecessors
    predecessors = predecessors_initialization(tasks, 0, 0)

    # Release tasks' common source nodes according to their periods
    # Deadlines are updated according to the corresponding release time
    # Real duration can be shorter than the wcet
    for t in range(0, max_time, scale):

        # The initial release for all tasks' common source nodes
        if t == 0:
            predecessors = predecessors_initialization(tasks, 0, 0)
            for tsk in range(len(tasks)):
                for vtx in range(tasks[tsk].V):
                    # release all the vertices without any precedences
                    if len(predecessors[tsk][vtx]) == 0:
                        # obtain the ready queue affinity
                        rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                        # generate the real execution time
                        if vtx == 0 or vtx == tasks[tsk].V-1:
                            rec = tasks[tsk].weights[vtx]
                            if rec != 0:
                                print("Node execution time error v0! ")
                        else:
                            # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                            rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)
                        sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, data_requests[tsk].req_data[vtx], rec, t, rec, tasks[tsk].deadlines[vtx], tasks[tsk].graph[vtx], rq_affinity, preempt_times)

                        rec_all += rec
                        # append to the corresponding ready queue
                        Ready_queues[rq_affinity].append(sub_job)

        # Release tasks' common source nodes according to their periods
        else:
            for tsk in range(len(tasks)):
                if t % tasks[tsk].period == 0:
                    # A new job of the task is released
                    # All these sub-jobs from previous period have to be cleaned

                    # remove the current running sub-jobs
                    for pro in range(total_processors):
                        if processors[pro].current_scheduling_vertex is not None:
                            if processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > t:
                                deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                        processors[pro].current_scheduling_vertex.vertex_id,
                                                        t, pro])
                            elif processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time == t:
                                processors[pro].schedule_record[-1].append(t)

                    # clean the ready queue
                    for q in range(len(Ready_queues)):
                        if len(Ready_queues[q]) > 0:
                            for v_id in range(len(Ready_queues[q])):
                                if Ready_queues[q][v_id].task_id == tsk:
                                    deadline_misses.append([tsk, Ready_queues[q][v_id].vertex_id, t])
                                    #print("new release: ", Ready_queues[q][v_id].task_id, Ready_queues[q][v_id].vertex_id, t, Ready_queues[q][v_id].pure_execution_time, Ready_queues[q][v_id].priority, Ready_queues[q][v_id].job_start_time, Ready_queues[q][v_id].deadline)

                            Ready_queues[q] = deque([vet for vet in Ready_queues[q] if vet.task_id != tsk])

                    # clean the predecessors
                    for p in range(len(predecessors[tsk])):
                        if len(predecessors[tsk][p]) > 0:
                            deadline_misses.append([tsk, p, t])

                    # initialize the predecessors of the corresponding task
                    predecessors[tsk] = predecessors_initialization(tasks, tsk, 1)
                    for vtx in range(tasks[tsk].V):
                        # release all the vertices without any precedences
                        if len(predecessors[tsk][vtx]) == 0:
                            # obtain the ready queue affinity
                            rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                            # generate the real execution time
                            if vtx == 0 or vtx == tasks[tsk].V - 1:
                                rec = tasks[tsk].weights[vtx]
                                if rec != 0:
                                    print("Node execution time error tp")
                            else:
                                # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                                rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)

                            sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, data_requests[tsk].req_data[vtx], rec, t,
                                             rec, tasks[tsk].deadlines[vtx] + t, tasks[tsk].graph[vtx],
                                             rq_affinity, preempt_times)
                            rec_all += rec
                            # append to the corresponding ready queue
                            Ready_queues[rq_affinity].append(sub_job)

        # Sort the corresponding ready queue according to their priorities
        for q in range(len(Ready_queues)):
            if len(Ready_queues[q]) > 0:
                Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: (x.priority, x.deadline)))

        current_time = t

        while(current_time < t + scale):
            # Check whether the scheduled vertex has finished its execution at the moment
            for pro in range(total_processors):

                if processors[pro].current_scheduling_vertex is not None:
                    # Deadline miss
                    if current_time > processors[pro].current_scheduling_vertex.deadline or (current_time == processors[pro].current_scheduling_vertex.deadline and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > current_time):
                        print("Current sub job misses its deadline, aborted! ", processors[pro].current_scheduling_vertex.task_id,
                                                processors[pro].current_scheduling_vertex.vertex_id, current_time)
                        # Abort the current vertex
                        deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                processors[pro].current_scheduling_vertex.vertex_id, current_time, pro])
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)

                        # Still in the same period
                        abt_tsk = processors[pro].current_scheduling_vertex
                        if current_time < abt_tsk.job_start_time + tasks[abt_tsk.task_id].period:
                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk, predecessors, Ready_queues, core_to_queue_mapping, affinities,
                                         data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 1)
                        # All its successors have missed their deadlines
                        else:
                            # do not need to update its predecessor
                            abt_all_successors = find_all_successors(tasks[abt_tsk.task_id].graph, abt_tsk.vertex_id)
                            deadline_misses.append([abt_tsk.task_id, abt_all_successors, current_time, pro])

                        processors[pro].current_scheduling_vertex = None

                    # The vertex has finished its execution
                    elif current_time == (processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time):
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)
                        # Update the precedence constraint
                        f_tsk_id = processors[pro].current_scheduling_vertex.task_id
                        f_vertex_id = processors[pro].current_scheduling_vertex.vertex_id
                        f_successors = processors[pro].current_scheduling_vertex.successors
                        # the finished vertex is not the common end node
                        if len(f_successors) != 0:
                            for f_v in range(len(f_successors)):
                                if f_vertex_id in predecessors[f_tsk_id][f_successors[f_v]]:
                                    predecessors[f_tsk_id][f_successors[f_v]].remove(f_vertex_id)
                                else:
                                    print("ERROR: The predecessor has been removed, Please check!", current_time, processors[pro].current_scheduling_vertex.job_start_time, tasks[f_tsk_id].weights[f_vertex_id], f_vertex_id, f_tsk_id, predecessors_initialization(tasks, f_tsk_id, 1)[f_successors[f_v]])
                                # If the successor has no remained constraint, it will be added into the corresponding ready queue
                                if len(predecessors[f_tsk_id][f_successors[f_v]]) == 0:
                                    # the successor's vertex id
                                    s_vertex_id = f_successors[f_v]
                                    # the start time of the whole job to calculate the relative deadline
                                    j_start_time = processors[pro].current_scheduling_vertex.job_start_time
                                    # obtain the ready queue affinity
                                    rq_affinity = core_to_queue_mapping[affinities[f_tsk_id][typed_info[f_tsk_id].typed[s_vertex_id]][0]]
                                    # generate the real execution time
                                    if s_vertex_id == 0 or s_vertex_id == tasks[f_tsk_id].V - 1:
                                        rec = tasks[f_tsk_id].weights[s_vertex_id]
                                        if rec != 0:
                                            print("Node execution time error ", rec, s_vertex_id, tasks[f_tsk_id].V)
                                    else:
                                        rec = gen_real_ec(tasks[f_tsk_id].weights[s_vertex_id], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)

                                    sub_job = Vertex(f_tsk_id, s_vertex_id, j_start_time, tasks[f_tsk_id].priority, data_requests[f_tsk_id].req_data[s_vertex_id], rec, current_time,
                                                 rec, tasks[f_tsk_id].deadlines[s_vertex_id] + j_start_time, tasks[f_tsk_id].graph[s_vertex_id],
                                                 rq_affinity, preempt_times)

                                    rec_all += rec
                                    # append to the corresponding ready queue
                                    Ready_queues[rq_affinity].append(sub_job)

                                    # Sort the corresponding ready queue according to their deadlines
                                    Ready_queues[rq_affinity] = deque(sorted(Ready_queues[rq_affinity], key=lambda x: (x.priority, x.deadline)))

                        # Set the current executing vertex in the processor as None
                        processors[pro].current_scheduling_vertex = None

            # List RM is applied, pickup the vertex from the ready queue with the highest priority (smallest period)
            # Either the previously executing vertex has been finished
            # Or the new released vertex can preempt the current executing vertex
            all_checked = [True] * len(Ready_queues)
            while any(all_checked):
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        if all_checked[q]:
                            all_checked[q] = False
                            single_queue_checked = False
                            while not single_queue_checked:
                                if len(Ready_queues[q]) <= 0:
                                    single_queue_checked = True
                                else:
                                    single_queue_checked = True
                                    sub_id = 0
                                    while sub_id < len(Ready_queues[q]):
                                        # it is impossible to be scheduled with the minimal data accessing time in the worst case
                                        # abort it in advance
                                        if l1_cache_time * (Ready_queues[q][sub_id].preempt_times + 1) + current_time + Ready_queues[q][sub_id].pure_execution_time > Ready_queues[q][sub_id].deadline:
                                            abt_tsk_rq = copy.deepcopy(Ready_queues[q][sub_id])
                                            del Ready_queues[q][sub_id]
                                            # print("ready queue checking: ", Ready_queues[q][sub_id].task_id, Ready_queues[q][sub_id].vertex_id, current_time, Ready_queues[q][sub_id].pure_execution_time, Ready_queues[q][sub_id].priority, Ready_queues[q][sub_id].job_start_time, Ready_queues[q][sub_id].deadline)
                                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk_rq, predecessors, Ready_queues, core_to_queue_mapping, affinities, data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 1)

                                            if len(new_ready_vertices) > 0:
                                                abt_tsk_id = abt_tsk_rq.task_id
                                                for s_id in range(len(new_ready_vertices)):
                                                    # The processors with the Ready_queues[rq_affinity] have to be checked again
                                                    rq_affinity = core_to_queue_mapping[affinities[abt_tsk_id][typed_info[abt_tsk_id].typed[new_ready_vertices[s_id]]][0]]

                                                    all_checked[rq_affinity] = True
                                                    if rq_affinity == q:
                                                        single_queue_checked = False
                                            sub_id = 0
                                        else:
                                            sub_id += 1
                    else:
                        all_checked[q] = False

            # Sort the corresponding ready queue according to their deadlines
            for q in range(len(Ready_queues)):
                if len(Ready_queues[q]) > 0:
                    Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: (x.priority, x.deadline)))

            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex == None and len(Ready_queues[processors[pro].ready_queue_id]) > 0:
                    # select the new task
                    n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                    n_tsk.start_time = current_time
                    data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data, memory_hierarchy)

                    if current_time + n_tsk.pure_execution_time + data_accessing_time > n_tsk.deadline:
                        print("The new scheduled sub job will miss its deadline. ",  n_tsk.task_id, n_tsk_vertex_id)
                    n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)
                    processors[pro].current_scheduling_vertex = n_tsk
                    # Record the starting corresponding processor's busy time
                    if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                        print("missed the finish time 1")
                    busy_start_temp = []
                    busy_start_temp.append(n_tsk.task_id)
                    busy_start_temp.append(n_tsk.vertex_id)
                    busy_start_temp.append(n_tsk.deadline)
                    busy_start_temp.append(n_tsk.rest_execution_time)
                    busy_start_temp.append(current_time)

                    processors[pro].schedule_record.append(busy_start_temp)

                if len(Ready_queues[processors[pro].ready_queue_id]) > 0 and processors[pro].current_scheduling_vertex is not None:
                    # if preemptive:
                    preemptable = False
                    # Check if the currently executing vertex can be preempted due to the FP-RM
                    if processors[pro].current_scheduling_vertex.preempt_times > 0:
                        if Ready_queues[processors[pro].ready_queue_id][0].priority < processors[pro].current_scheduling_vertex.priority:
                            preemptable = True
                        if Ready_queues[processors[pro].ready_queue_id][0].priority == processors[pro].current_scheduling_vertex.priority and Ready_queues[processors[pro].ready_queue_id][0].deadline < processors[pro].current_scheduling_vertex.deadline:
                            preemptable = True

                    if preemptable:
                        # schedule the new vertex with the highest priority
                        n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                        # add the data accessing time
                        data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data,
                                                                                         memory_hierarchy)
                        n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)

                        # modify the preempted vertex
                        preempted_tsk = processors[pro].current_scheduling_vertex
                        # check if the vertex is preempted when accessing data
                        if current_time - preempted_tsk.start_time > preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time:
                            preempted_tsk.pure_execution_time = preempted_tsk.pure_execution_time - (current_time - preempted_tsk.start_time - (preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time))
                        if preempted_tsk.pure_execution_time < 0:
                            print("Not preemptable sub job is preempted! ")

                        if current_time + n_tsk.rest_execution_time > n_tsk.deadline:
                            print("The preempting sub job will miss its deadline. ", n_tsk.task_id, n_tsk.vertex_id)
                        # update the available preemptable times
                        preempted_tsk.preempt_times -= 1
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)
                        # Append the preempted vertex back and sort the corresponding ready queue
                        Ready_queues[processors[pro].ready_queue_id].append(preempted_tsk)
                        Ready_queues[processors[pro].ready_queue_id] = deque(sorted(Ready_queues[processors[pro].ready_queue_id], key=lambda x: (x.priority, x.deadline)))

                        n_tsk.start_time = current_time
                        processors[pro].current_scheduling_vertex = n_tsk
                        # Record the starting corresponding processor's busy time
                        if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                            print("missed the finish time")
                        busy_start_temp = []
                        busy_start_temp.append(n_tsk.task_id)
                        busy_start_temp.append(n_tsk.vertex_id)
                        busy_start_temp.append(n_tsk.deadline)
                        busy_start_temp.append(n_tsk.rest_execution_time)
                        busy_start_temp.append(current_time)
                        processors[pro].schedule_record.append(busy_start_temp)

            # Update the current time for next check
            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex is not None:
                    current_time_temp[pro] = min(processors[pro].current_scheduling_vertex.deadline, processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time)
                    # current_time_temp[pro] = processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time
                else:
                    current_time_temp[pro] = t + scale

            current_time = min(current_time_temp)

            # The next checking time is out of the range of maximum simulating time
            # Stop the simulation
            if current_time >= max_time:
                current_time = max_time
                for pro in range(total_processors):
                    if processors[pro].current_scheduling_vertex is not None:
                        processors[pro].schedule_record[-1].append(current_time)
                        if current_time > processors[pro].current_scheduling_vertex.deadline or current_time > processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time:
                            # deadline miss
                            deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                    processors[pro].current_scheduling_vertex.vertex_id, current_time,
                                                    pro])
                        elif current_time < processors[pro].current_scheduling_vertex.deadline:
                            print("Maximum simulation time is set incorrectly")

                # clean all the ready queues:
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        for v_id in range(len(Ready_queues[q])):
                            deadline_misses.append([Ready_queues[q][v_id].task_id, Ready_queues[q][v_id].vertex_id, t])
                    Ready_queues[q].clear()

    #for i, processor in enumerate(processors):
    #    for j, task_info in enumerate(processor.schedule_record):
    #        if len(task_info) != 6:
    #            print (task_info)

    print("Average released utilization: ", rec_all/total_processors/max_time)

    return processors, deadline_misses


# The general Typed DAG schedule simulator with EDF scheduling algorithm
# The processor allocations, i.e., affinities, are given by different federated scheduling algorithms
# Discard version: sub-jobs that can potentially miss their deadlines are discarded in advance,
# i.e., deleted by the ready queue checker, and do not be scheduled.
def typed_dag_schedule_edf_discard_sim(tasks_org, affinities_org, typed_org, data_requests, total_processors, max_time, scale, preempt_times, memory_org, main_mem_time, avg_ratio, min_ratio, std_dev):
    tasks = copy.deepcopy(tasks_org)
    affinities = copy.deepcopy(affinities_org)
    typed_info = copy.deepcopy(typed_org)
    data_requests = copy.deepcopy(data_requests)
    memory_hierarchy = copy.deepcopy(memory_org)

    rec_all = 0

    # record all the possible deadline misses
    deadline_misses = []

    # Initialize the ready queues and find the core to ready queue mapping
    Ready_queues, core_to_queue_mapping = ready_queues_initialization(affinities, total_processors)

    # Initialize all the available processors
    processors = []
    print(total_processors, len(core_to_queue_mapping))
    for i in range(total_processors):
        processors.append(Processor(core_to_queue_mapping[i]))
        processors[i].current_scheduling_vertex = None

    # Check the next check time for each cluster
    current_time_temp = []
    # initialize the current time temp
    for pro in range(total_processors):
        current_time_temp.append(scale)

    # Initialize the predecessors
    predecessors = predecessors_initialization(tasks, 0, 0)

    # Release tasks' common source nodes according to their periods
    # Deadlines are updated according to the corresponding release time
    # Real duration can be shorter than the wcet
    for t in range(0, max_time, scale):

        # The initial release for all tasks' common source nodes
        if t == 0:
            predecessors = predecessors_initialization(tasks, 0, 0)
            for tsk in range(len(tasks)):
                for vtx in range(tasks[tsk].V):
                    # release all the vertices without any precedences
                    if len(predecessors[tsk][vtx]) == 0:
                        # obtain the ready queue affinity
                        rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                        # generate the real execution time
                        if vtx == 0 or vtx == tasks[tsk].V-1:
                            rec = tasks[tsk].weights[vtx]
                            if rec != 0:
                                print("Node execution time error v0! ")
                        else:
                            # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                            rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)
                        sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, data_requests[tsk].req_data[vtx], rec, t, rec, tasks[tsk].deadlines[vtx], tasks[tsk].graph[vtx], rq_affinity, preempt_times)

                        rec_all += rec
                        # append to the corresponding ready queue
                        Ready_queues[rq_affinity].append(sub_job)

        # Release tasks' common source nodes according to their periods
        else:
            for tsk in range(len(tasks)):
                if t % tasks[tsk].period == 0:
                    # A new job of the task is released
                    # All these sub-jobs from previous period have to be cleaned

                    # remove the current running sub-jobs
                    for pro in range(total_processors):
                        if processors[pro].current_scheduling_vertex is not None:
                            if processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > t:
                                deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                        processors[pro].current_scheduling_vertex.vertex_id,
                                                        t, pro])
                            elif processors[pro].current_scheduling_vertex.task_id == tsk and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time == t:
                                processors[pro].schedule_record[-1].append(t)

                    # clean the ready queue
                    for q in range(len(Ready_queues)):
                        if len(Ready_queues[q]) > 0:
                            for v_id in range(len(Ready_queues[q])):
                                if Ready_queues[q][v_id].task_id == tsk:
                                    deadline_misses.append([tsk, Ready_queues[q][v_id].vertex_id, t])
                                    #print("new release: ", Ready_queues[q][v_id].task_id, Ready_queues[q][v_id].vertex_id, t, Ready_queues[q][v_id].pure_execution_time, Ready_queues[q][v_id].priority, Ready_queues[q][v_id].job_start_time, Ready_queues[q][v_id].deadline)

                            Ready_queues[q] = deque([vet for vet in Ready_queues[q] if vet.task_id != tsk])

                    # clean the predecessors
                    for p in range(len(predecessors[tsk])):
                        if len(predecessors[tsk][p]) > 0:
                            deadline_misses.append([tsk, p, t])

                    # initialize the predecessors of the corresponding task
                    predecessors[tsk] = predecessors_initialization(tasks, tsk, 1)
                    for vtx in range(tasks[tsk].V):
                        # release all the vertices without any precedences
                        if len(predecessors[tsk][vtx]) == 0:
                            # obtain the ready queue affinity
                            rq_affinity = core_to_queue_mapping[affinities[tsk][typed_info[tsk].typed[vtx]][0]]
                            # generate the real execution time
                            if vtx == 0 or vtx == tasks[tsk].V - 1:
                                rec = tasks[tsk].weights[vtx]
                                if rec != 0:
                                    print("Node execution time error tp")
                            else:
                                # rec = tasks[tsk].weights[vtx] - ((preempt_times + 1) * main_mem_time)
                                rec = gen_real_ec(tasks[tsk].weights[vtx], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)

                            sub_job = Vertex(tsk, vtx, t, tasks[tsk].priority, data_requests[tsk].req_data[vtx], rec, t,
                                             rec, tasks[tsk].deadlines[vtx] + t, tasks[tsk].graph[vtx],
                                             rq_affinity, preempt_times)
                            rec_all += rec
                            # append to the corresponding ready queue
                            Ready_queues[rq_affinity].append(sub_job)

        # Sort the corresponding ready queue according to their priorities
        for q in range(len(Ready_queues)):
            if len(Ready_queues[q]) > 0:
                Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: x.deadline))

        current_time = t

        while(current_time < t + scale):
            # Check whether the scheduled vertex has finished its execution at the moment
            for pro in range(total_processors):

                if processors[pro].current_scheduling_vertex is not None:
                    # Deadline miss
                    if current_time > processors[pro].current_scheduling_vertex.deadline or (current_time == processors[pro].current_scheduling_vertex.deadline and processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time > current_time):
                        print("Deadline miss task is scheduled, please check!")
                        # Abort the current vertex
                        deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                processors[pro].current_scheduling_vertex.vertex_id, current_time, pro])
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)

                        # Still in the same period
                        if current_time < processors[pro].current_scheduling_vertex.job_start_time + tasks[processors[pro].current_scheduling_vertex.task_id].period:
                            abt_tsk = processors[pro].current_scheduling_vertex
                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk, predecessors, Ready_queues, core_to_queue_mapping, affinities,
                                         data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 0)
                        # All its successors have missed their deadlines
                        else:
                            # do not need to update its predecessor
                            abt_all_successors = find_all_successors(tasks[abt_tsk.task_id].graph, abt_tsk.vertex_id)
                            deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                    abt_all_successors, current_time,
                                                    pro])

                        processors[pro].current_scheduling_vertex = None

                    # The vertex has finished its execution
                    elif current_time == (processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time):
                        # Record the end corresponding processor's busy time
                        processors[pro].schedule_record[-1].append(current_time)
                        # Update the precedence constraint
                        f_tsk_id = processors[pro].current_scheduling_vertex.task_id
                        f_vertex_id = processors[pro].current_scheduling_vertex.vertex_id
                        f_successors = processors[pro].current_scheduling_vertex.successors
                        # the finished vertex is not the common end node
                        if len(f_successors) != 0:
                            for f_v in range(len(f_successors)):
                                if f_vertex_id in predecessors[f_tsk_id][f_successors[f_v]]:
                                    predecessors[f_tsk_id][f_successors[f_v]].remove(f_vertex_id)
                                else:
                                    print("ERROR: The predecessor has been removed, Please check!", current_time, processors[pro].current_scheduling_vertex.job_start_time, tasks[f_tsk_id].weights[f_vertex_id], f_vertex_id, f_tsk_id, predecessors_initialization(tasks, f_tsk_id, 1)[f_successors[f_v]])
                                # If the successor has no remained constraint, it will be added into the corresponding ready queue
                                if len(predecessors[f_tsk_id][f_successors[f_v]]) == 0:
                                    # the successor's vertex id
                                    s_vertex_id = f_successors[f_v]
                                    # the start time of the whole job to calculate the relative deadline
                                    j_start_time = processors[pro].current_scheduling_vertex.job_start_time
                                    # obtain the ready queue affinity
                                    rq_affinity = core_to_queue_mapping[affinities[f_tsk_id][typed_info[f_tsk_id].typed[s_vertex_id]][0]]
                                    # generate the real execution time
                                    if s_vertex_id == 0 or s_vertex_id == tasks[f_tsk_id].V - 1:
                                        rec = tasks[f_tsk_id].weights[s_vertex_id]
                                        if rec != 0:
                                            print("Node execution time error ", rec, s_vertex_id, tasks[f_tsk_id].V)
                                    else:
                                        # rec = tasks[f_tsk_id].weights[s_vertex_id] - ((preempt_times + 1) * main_mem_time)
                                        if tasks[f_tsk_id].weights[s_vertex_id] < 0:
                                            print("Weight is wrong!")
                                        rec = gen_real_ec(tasks[f_tsk_id].weights[s_vertex_id], preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev)
                                        if rec < 0:
                                            print("Execution time error! ", rec)
                                            return

                                    if rec > tasks[f_tsk_id].weights[s_vertex_id]:
                                        print("Execution time error rec")

                                    sub_job = Vertex(f_tsk_id, s_vertex_id, j_start_time, tasks[f_tsk_id].priority, data_requests[f_tsk_id].req_data[s_vertex_id], rec, current_time,
                                                 rec, tasks[f_tsk_id].deadlines[s_vertex_id] + j_start_time, tasks[f_tsk_id].graph[s_vertex_id],
                                                 rq_affinity, preempt_times)

                                    rec_all += rec
                                    # append to the corresponding ready queue
                                    Ready_queues[rq_affinity].append(sub_job)

                                    # Sort the corresponding ready queue according to their deadlines
                                    Ready_queues[rq_affinity] = deque(sorted(Ready_queues[rq_affinity], key=lambda x: x.deadline))

                        # Set the current executing vertex in the processor as None
                        processors[pro].current_scheduling_vertex = None

            # List RM is applied, pickup the vertex from the ready queue with the highest priority (smallest period)
            # Either the previously executing vertex has been finished
            # Or the new released vertex can preempt the current executing vertex
            all_checked = [True] * len(Ready_queues)
            while any(all_checked):
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        if all_checked[q]:
                            all_checked[q] = False
                            single_queue_checked = False
                            while not single_queue_checked:
                                if len(Ready_queues[q]) <= 0:
                                    single_queue_checked = True
                                else:
                                    single_queue_checked = True
                                    sub_id = 0
                                    while sub_id < len(Ready_queues[q]):
                                        # it is impossible to be scheduled with the minimal data accessing time in the worst case
                                        # abort it in advance
                                        if main_mem_time * (Ready_queues[q][sub_id].preempt_times + 1) + current_time + Ready_queues[q][sub_id].pure_execution_time > Ready_queues[q][sub_id].deadline:
                                            abt_tsk_rq = copy.deepcopy(Ready_queues[q][sub_id])
                                            del Ready_queues[q][sub_id]
                                            # print("ready queue checking: ", Ready_queues[q][sub_id].task_id, Ready_queues[q][sub_id].vertex_id, current_time, Ready_queues[q][sub_id].pure_execution_time, Ready_queues[q][sub_id].priority, Ready_queues[q][sub_id].job_start_time, Ready_queues[q][sub_id].deadline)
                                            predecessors, Ready_queues, rec_all, new_ready_vertices = abort_vertex(tasks, abt_tsk_rq, predecessors, Ready_queues, core_to_queue_mapping, affinities, data_requests, typed_info, current_time, preempt_times, main_mem_time, min_ratio, avg_ratio, std_dev, rec_all, 0)

                                            if len(new_ready_vertices) > 0:
                                                abt_tsk_id = abt_tsk_rq.task_id
                                                for s_id in range(len(new_ready_vertices)):
                                                    # The processors with the Ready_queues[rq_affinity] have to be checked again
                                                    rq_affinity = core_to_queue_mapping[affinities[abt_tsk_id][typed_info[abt_tsk_id].typed[new_ready_vertices[s_id]]][0]]

                                                    all_checked[rq_affinity] = True
                                                    if rq_affinity == q:
                                                        single_queue_checked = False
                                            sub_id = 0
                                        else:
                                            sub_id += 1
                    else:
                        all_checked[q] = False

            # Sort the corresponding ready queue according to their deadlines
            for q in range(len(Ready_queues)):
                if len(Ready_queues[q]) > 0:
                    Ready_queues[q] = deque(sorted(Ready_queues[q], key=lambda x: x.deadline))

            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex == None and len(Ready_queues[processors[pro].ready_queue_id]) > 0:
                    # select the new task
                    n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                    n_tsk.start_time = current_time
                    data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data, memory_hierarchy)
                    # The new vertex can potentially finish its execution
                    if current_time + n_tsk.pure_execution_time + data_accessing_time <= n_tsk.deadline:
                        n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)
                        processors[pro].current_scheduling_vertex = n_tsk
                        # Record the starting corresponding processor's busy time
                        if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                            print("missed the finish time 1")
                        busy_start_temp = []
                        busy_start_temp.append(n_tsk.task_id)
                        busy_start_temp.append(n_tsk.vertex_id)
                        busy_start_temp.append(n_tsk.deadline)
                        busy_start_temp.append(n_tsk.rest_execution_time)
                        busy_start_temp.append(current_time)

                        processors[pro].schedule_record.append(busy_start_temp)
                    # The new vertex will miss the deadline any way
                    else:
                        print("Ready queue check is wrong, the sub-job in the ready queue is not schedulable!", n_tsk.task_id, n_tsk.vertex_id, current_time, n_tsk.pure_execution_time, data_accessing_time, n_tsk.job_start_time, n_tsk.priority, n_tsk.deadline)

                if len(Ready_queues[processors[pro].ready_queue_id]) > 0 and processors[pro].current_scheduling_vertex is not None:
                    # if preemptive:
                    preemptable = False
                    # Check if the currently executing vertex can be preempted due to the FP-RM
                    if processors[pro].current_scheduling_vertex.preempt_times > 0:
                        if Ready_queues[processors[pro].ready_queue_id][0].priority < processors[pro].current_scheduling_vertex.priority:
                            preemptable = True
                        if Ready_queues[processors[pro].ready_queue_id][0].priority == processors[pro].current_scheduling_vertex.priority and Ready_queues[processors[pro].ready_queue_id][0].deadline < processors[pro].current_scheduling_vertex.deadline:
                            preemptable = True

                    if preemptable:
                        # schedule the new vertex with the highest priority
                        n_tsk = Ready_queues[processors[pro].ready_queue_id].popleft()
                        # add the data accessing time
                        data_accessing_time, memory_hierarchy = data_access_time(n_tsk.requested_data,
                                                                                         memory_hierarchy)
                        n_tsk.rest_execution_time = copy.deepcopy(n_tsk.pure_execution_time) + copy.deepcopy(data_accessing_time)

                        # The new vertex can potentially finish its execution
                        if current_time + n_tsk.rest_execution_time <= n_tsk.deadline:
                            # modify the preempted vertex
                            preempted_tsk = processors[pro].current_scheduling_vertex
                            # check if the vertex is preempted when accessing data
                            if current_time - preempted_tsk.start_time > preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time:
                                preempted_tsk.pure_execution_time = preempted_tsk.pure_execution_time - (current_time - preempted_tsk.start_time - (preempted_tsk.rest_execution_time - preempted_tsk.pure_execution_time))
                            if preempted_tsk.pure_execution_time < 0:
                                print("Not preemptable sub job is preempted! ")

                            # update the available preemptable times
                            preempted_tsk.preempt_times -= 1
                            # Record the end corresponding processor's busy time
                            processors[pro].schedule_record[-1].append(current_time)
                            # Append the preempted vertex back and sort the corresponding ready queue
                            Ready_queues[processors[pro].ready_queue_id].append(preempted_tsk)
                            Ready_queues[processors[pro].ready_queue_id] = deque(sorted(Ready_queues[processors[pro].ready_queue_id], key=lambda x: x.deadline))

                            n_tsk.start_time = current_time
                            processors[pro].current_scheduling_vertex = n_tsk
                            # Record the starting corresponding processor's busy time
                            if len(processors[pro].schedule_record) >= 1 and len(processors[pro].schedule_record[-1]) < 6:
                                print("missed the finish time")
                            busy_start_temp = []
                            busy_start_temp.append(n_tsk.task_id)
                            busy_start_temp.append(n_tsk.vertex_id)
                            busy_start_temp.append(n_tsk.deadline)
                            busy_start_temp.append(n_tsk.rest_execution_time)
                            busy_start_temp.append(current_time)
                            processors[pro].schedule_record.append(busy_start_temp)
                        else:
                            print("Wrong job in the ready queue! Preempt the current executing sub-job!")

            # Update the current time for next check
            for pro in range(total_processors):
                if processors[pro].current_scheduling_vertex is not None:
                    # FIXME: Check the min(deadline, possible finish time) rather than only the possible finish time
                    # to make sure that the following vertices can potentially finish their execution
                    if processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time > processors[pro].current_scheduling_vertex.deadline:
                        print("Some thing wrong, wrong vertex is scheduled !", processors[pro].current_scheduling_vertex.task_id, processors[pro].current_scheduling_vertex.vertex_id, processors[pro].current_scheduling_vertex.deadline)
                    current_time_temp[pro] = min(processors[pro].current_scheduling_vertex.deadline, processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time)
                    # current_time_temp[pro] = processors[pro].current_scheduling_vertex.rest_execution_time + processors[pro].current_scheduling_vertex.start_time
                else:
                    current_time_temp[pro] = t + scale

            current_time = min(current_time_temp)

            # The next checking time is out of the range of maximum simulating time
            # Stop the simulation
            if current_time >= max_time:
                current_time = max_time
                for pro in range(total_processors):
                    if processors[pro].current_scheduling_vertex is not None:
                        processors[pro].schedule_record[-1].append(current_time)
                        if current_time > processors[pro].current_scheduling_vertex.deadline or current_time > processors[pro].current_scheduling_vertex.start_time + processors[pro].current_scheduling_vertex.rest_execution_time:
                            # deadline miss
                            deadline_misses.append([processors[pro].current_scheduling_vertex.task_id,
                                                    processors[pro].current_scheduling_vertex.vertex_id, current_time,
                                                    pro])
                        elif current_time < processors[pro].current_scheduling_vertex.deadline:
                            print("Maximum simulation time is set incorrectly")

                # clean all the ready queues:
                for q in range(len(Ready_queues)):
                    if len(Ready_queues[q]) > 0:
                        for v_id in range(len(Ready_queues[q])):
                            deadline_misses.append([Ready_queues[q][v_id].task_id, Ready_queues[q][v_id].vertex_id, t])
                    Ready_queues[q].clear()

    print("Average released utilization: ", rec_all/max_time)

    return processors, deadline_misses


# Generate the schedule for task sets according to different approaches
def typed_dag_schedule_gen(tasksets_org, typed_org, data_requests, affinities, processor_a, processor_b, max_time, scale, preempt_times, main_mem_size, main_mem_time, fast_mem_size, fast_mem_time, l1_cache_size, l1_cache_time, avg_ratio, min_ratio, std_dev):
    tasksets = copy.deepcopy(tasksets_org)
    typed_info = copy.deepcopy(typed_org)

    processors_info = []
    deadline_miss_info = []

    total_processors = processor_a + processor_b


    periods = []
    utilization_set = 0
    for tsk in range(len(tasksets)):
        periods.append(tasksets[tsk].period)
        utilization_set += tasksets[tsk].utilization
    print("Designed utilization: ", utilization_set)

    hyper_period = misc.find_lcm(periods)

    memory_org = mem.memory_initialization(l1_cache_size, l1_cache_time, fast_mem_size, fast_mem_time, main_mem_size, main_mem_time, data_requests, hyper_period, periods, 0)

    processors_info, deadline_miss_info = typed_dag_schedule_rm_discard_sim(tasksets, affinities, typed_info, data_requests, total_processors, max_time, scale, preempt_times, memory_org, main_mem_time, avg_ratio, min_ratio, std_dev)


    durations = 0
    for i, processor in enumerate(processors_info):
        for j, task_info in enumerate(processor.schedule_record):
            # task_id, start_time, finish_time = task_info
            task_id, vertex_id, deadline, rest_execution, start_time, finish_time = task_info
            # print(i, task_id, start_time, finish_time)
            if finish_time > max_time:
                print("Finish time is higher than the maximum evaluation time! ")
            if finish_time > deadline:
                print("Finish time is higher than the deadline! ", task_id, vertex_id, deadline, rest_execution, start_time, finish_time)
            duration = finish_time - start_time
            durations += duration

    print("Actual total utilization: ", durations/max_time)
    print("Actual per core utilization: ", durations / max_time / total_processors)


    return processors_info, deadline_miss_info
