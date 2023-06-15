from __future__ import division
import numpy as np
import os
import math
import sys
import getopt
import copy
from typing import *
import time
import json
from operator import itemgetter
from collections import defaultdict
import generator_pure_dict as gen
import data_requests as data_req
import typed_core_allocation as typed
import read_configuration as readf


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


# The dictionary of requested data of each task
class ReqData:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.req_data = defaultdict(hex)  # the requested data for each vertex


class Type:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.typed = defaultdict(int) # the type of processors allocation, 1: type A; 2: type B; 0: undefined
        self.utilizationA = float # the utilization of all vertices in type A cores
        self.utilizationB = float # the utilization of all vertices in type B cores
        self.cpA = int  # critical path of all vertices in type A cores
        self.cpB = int  # critical path of all vertices in type B cores


# {
#   "tasks": [
#     {
#       "task_id": 1,
#       "period": 100,
#       "deadline": 50,
#       "vertices": [
#         {
#           "vertex_id": 1,
#           "execution_time": 10,
#           "successors": [2],
#           "requested_data_address": '0x14',
#           "core_type" : 1
#         },
#         {
#           "vertex_id": 2,
#           "execution_time": 15,
#           "successors": [3],
#           "requested_data_address": '0x14',
#           "core_type" : 1
#         },
#         {
#           "vertex_id": 3,
#           "execution_time": 20,
#           "successors": [],
#           "requested_data_address": '0x14',
#           "core_type" : 1
#         }
#       ]
#     },
#     {
#       "task_id": 2,
#       "period": 200,
#       "deadline": 80,
#       "vertices": [
#         {
#           "vertex_id": 1,
#           "execution_time": 12,
#           "successors": [2],
#           "requested_data_address": '0x14',
#           "core_type" : 1
#         },
#         {
#           "vertex_id": 2,
#           "execution_time": 8,
#           "successors": [1],
#           "requested_data_address": '0x14',
#           "core_type" : 2
#         }
#       ]
#     }
#   ]
# }




def main(argv):

    tskset_file_name = 'taskset.json'
    conf_file_name = 'configure.json'

    try:
        opts, args = getopt.getopt(argv, "hi:", ["tskfname"])
    except getopt.GetoptError:
        print('tasksets_input_convertor.py -i <the JSON task set file name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('tasksets_input_convertor.py -i <the JSON task set file name>')
            sys.exit()
        elif opt in ("-i", "--tskfname"):
            tskset_file_name = str(arg)

    print('Read configurations . . .')

    conf = readf.read_conf(conf_file_name)
    msets = conf['mset'][0]
    ntasks = conf['ntasks'][0]
    num_nodes = conf['nnodes']
    processor_a = conf['aprocessor'][0]
    processor_b = conf['bprocessor'][0]
    pc_prob = conf['pr_prob']
    sparse = conf['sparse'][0]
    util_all = conf['utilization']
    preempt_times = conf['preempt_times'][0]
    scale = conf['scale'][0]
    main_mem_time = conf['main_mem_time'][0]

    num_data_all = conf['num_data_all'][0]
    num_freq_data = conf['num_freq_data'][0]
    percent_freq = conf['percent_freq'][0]
    allow_freq = conf['allow_freq'][0]

    skewness = conf['skewness'][0]
    per_heavy = conf['per_heavy'][0]
    one_type_only = conf['one_type_only'][0]

    if os.path.exists(tskset_file_name):
        with open(tskset_file_name, 'r') as json_file:
            data = json.load(json_file)
    else:
        print("Please enter correct json file name of task sets.")
        return


    print('Task set converting . . .')
    for ut in range(len(util_all)):
        print('Converting task set with utilization: ', util_all[ut])

        utili = float(util_all[ut] / 100)
        utilization = utili*(processor_a + processor_b)
        tasksets_name = '../experiments/inputs/tasks_pure/tasksets_pure_' + str(msets) + '_' + str(ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(main_mem_time) + '.npy'

        tasksets_data_name = '../experiments/inputs/tasks_data_request/tasksets_data_req_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
            preempt_times) + '_m' + str(main_mem_time) + '_d' + str(num_data_all) + '_' + str(
            num_freq_data) + '_' + str(percent_freq) + '_' + str(allow_freq) + '.npy'

        tasksets_typed_name = '../experiments/inputs/tasks_typed/tasksets_typed_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
            preempt_times) + '_m' + str(main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
            one_type_only) + '.npy'


        tasksets = []
        tasksets_typed = []
        tasksets_data = []
        if msets > 1:
            print("Please note, this script can only handle one set at a time!")
            return

        for i in range(msets):
            taskset = []
            taskset_typed = []
            taskset_data = []

            tasks = data["tasks"]


            for j in range(ntasks):
                data_task = tasks[j]
                num_vertices = len(data_task["vertices"])
                num_nodes = num_vertices + 2
                vertices = data_task["vertices"]
                # DAG task pure
                tsk_temp = Graph(num_nodes)
                # data
                requested_data = ReqData(num_nodes)
                # typed
                type_temp = Type(num_nodes)

                tsk_temp.tsk_id = data_task["task_id"]
                tsk_temp.period = data_task["period"] * scale
                tsk_temp.deadline = data_task["deadline"] * scale

                WCETs_float = []
                utilis = []
                # add the common source node
                WCETs_float.append(0)
                struct = defaultdict(list)
                struct[0].append(1)
                utilis.append(0)
                requested_data.req_data[0] = hex(0)
                type_temp.typed[0] = 1

                for n in range(num_vertices):
                    WCETs_float.append(vertices[n]["execution_time"]*scale)
                    struct[n+1] = vertices[n]["successors"]
                    utilis.append(vertices[n]["execution_time"]*scale/tsk_temp.period)
                    requested_data.req_data[n+1] = vertices[n]["requested_data_address"]
                    type_temp.typed[n+1] = vertices[n]["core_type"]
                # add the common end node
                WCETs_float.append(0)
                utilis.append(0)
                tsk_temp.utilizations = utilis
                tsk_temp.utilization = sum(tsk_temp.utilizations)
                struct[num_vertices].append(num_vertices + 1)
                requested_data.req_data[num_nodes-1] = hex(0)
                type_temp.typed[num_nodes-1] = 1

                WCETs_int = list(map(int, WCETs_float))
                tsk_temp.weights = WCETs_int
                tsk_temp.priority = [0] * (num_nodes)

                # check the connection to common source node
                all_list = range(1, num_nodes - 1)
                set_all_list = set(all_list)
                for node in range(1, num_nodes - 1):
                    set_all_list = set_all_list - set(struct[node])

                not_connected_source = list(set_all_list)
                if len(not_connected_source) > 0:
                    for node in range(len(not_connected_source)):
                        struct[0].append(not_connected_source[node])

                # check the connection to common end node
                for node in range(1, num_nodes - 1):
                    if len(struct[node]) == 0:
                        struct[node].append(num_nodes - 1)
                    if len(struct[node]) < 1:
                        print('Length3 : ', len(struct[node]))
                        print('something wrong 0!!!')

                # append the last item
                struct[num_nodes - 1]

                tsk_temp.graph = copy.deepcopy(dict(sorted(struct.items())))

                tsk_temp.predecessors = gen.find_predecessors(tsk_temp.graph, num_nodes)

                longest_path = gen.find_longest_path(tsk_temp.graph, tsk_temp.weights)[1]
                if longest_path <= tsk_temp.period:
                    tsk_temp.cp = int(longest_path)
                    tsk_temp.deadlines = gen.calculate_deadlines(tsk_temp.graph, tsk_temp.weights, tsk_temp.deadline)
                    taskset.append(tsk_temp)
                else:
                    print("The task is infeasible by default, please check")
                    return

                taskset_data.append(requested_data)



                ################################
                # handle the typed infomation
                # temp weights for vertices only in processor A
                temp_weights_a = copy.deepcopy(tsk_temp.weights)
                # temp weights for vertices only in processor B
                temp_weights_b = copy.deepcopy(tsk_temp.weights)

                for node in range(0, num_nodes):
                    if type_temp.typed[node] == 1:
                        temp_weights_b[node] = 0

                    else:
                        temp_weights_a[node] = 0

                # In the following some additional information is calculated for future usage
                # calculate the L_i^a and L_i^b
                La = gen.find_longest_path(tsk_temp.graph, temp_weights_a)[1]
                Lb = gen.find_longest_path(tsk_temp.graph, temp_weights_b)[1]

                if La == tsk_temp.period / 3 or La == tsk_temp.period / 2:
                    La += 1
                if Lb == tsk_temp.period / 3 or Lb == tsk_temp.period / 2:
                    Lb += 1

                type_temp.cpA = La
                type_temp.cpB = Lb

                # Accumulative execution time of vertices on type A and B
                type_temp.utilizationA = sum(temp_weights_a) / tsk_temp.period
                type_temp.utilizationB = sum(temp_weights_b) / tsk_temp.period

                # Check the common source and end node
                # if they are allocated to the wrong typed of processor
                if type_temp.utilizationA == 0:
                    # no A core is required
                    type_temp.typed[0] = 2
                    type_temp.typed[num_nodes - 1] = 2

                if type_temp.utilizationB == 0:
                    # no B core is required
                    type_temp.typed[0] = 1
                    type_temp.typed[num_nodes - 1] = 1

                taskset_typed.append(type_temp)

            taskset_temp = copy.deepcopy(taskset)
            sorted_index = [i[0] for i in
                            sorted(enumerate(taskset_temp), key=lambda x: (x[1].period, -x[1].utilization))]
            for tsk in range(ntasks):
                taskset[sorted_index[tsk]].priority = int(tsk + 1)


            tasksets.append(taskset)
            tasksets_typed.append(taskset_typed)
            tasksets_data.append(taskset_data)


        np.save(tasksets_name, np.array(tasksets, dtype=object))
        np.save(tasksets_typed_name, np.array(tasksets_typed, dtype=object))
        np.save(tasksets_data_name, np.array(tasksets_data, dtype=object))



if __name__ == "__main__":
    main(sys.argv[1:])
