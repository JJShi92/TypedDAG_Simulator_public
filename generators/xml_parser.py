import xml.etree.ElementTree as ET
import numpy as np
import os
import math
import sys
import getopt
import copy
from drs import drs
from typing import *
import time
import json
from operator import itemgetter
from collections import defaultdict
import generator_pure_dict as gen
import data_requests as data_req
import typed_core_allocation as typed
import read_configuration as readf

# Replace 'spectestdata.xml' with the actual filename
xml_file = 'spectestdata.xml'


def count_resources(xml_file, resource_type):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    count = 0

    for resource_elem in root.findall(".//{http://opendse.sourceforge.net}resource"):
        subtype_elem = resource_elem.find(".//{http://opendse.sourceforge.net}attribute[@name='SUBTYPE']")
        if subtype_elem is not None and subtype_elem.text == resource_type:
            count += 1

    return count


core_count = count_resources(xml_file, "core")
accelerator_count = count_resources(xml_file, "accelerator")

print("Number of cores: ", core_count)
print("Number of accelerators: ", accelerator_count)


class Graph_xml:
    def __init__(self):
        self.graph = defaultdict(list)  # default dictionary to store graph

    def addEdge(self, u, v):
        self.graph[u].append(v)


def get_graph(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <application> element
    application = root.find('{http://opendse.sourceforge.net}application')

    # Collect dependencies and put tasks into sets
    sets = []
    all_tasks = set()
    for dependency in application.iter('{http://opendse.sourceforge.net}dependency'):
        source = dependency.get('source')
        destination = dependency.get('destination')

        if 'data' not in source and 'data' not in destination:
            all_tasks.update([source, destination])
            relevant_sets = [s for s in sets if source in s or destination in s]
            if relevant_sets:
                merged_set = set().union(*relevant_sets)
                merged_set.add(source)
                merged_set.add(destination)
                sets = [s for s in sets if s not in relevant_sets]  # remove merged sets from list
                sets.append(merged_set)  # add merged set to list
            else:
                new_set = {source, destination}
                sets.append(new_set)

    # Create a mapping of old task id to new task id
    task_mapping = {old: new for new, old in enumerate(sorted(list(all_tasks)), start=1)}
    print("Task Mapping: ", task_mapping)

    # Create a separate Graph for each set
    graphs = []
    for s in sets:
        g = Graph_xml()
        for dependency in application.iter('{http://opendse.sourceforge.net}dependency'):
            source = dependency.get('source')
            destination = dependency.get('destination')

            if 'data' not in source and 'data' not in destination and source in s and destination in s:
                # Replace old task ids with new task ids
                g.addEdge(task_mapping[source], task_mapping[destination])

        graphs.append(g)

    # Add common source and destination nodes
    num_tasks = len(all_tasks)
    '''
    for g in graphs:
        for node in range(1, num_tasks + 1):
            if node not in g.graph:
                g.addEdge(0, node)  # add edge from common source to task
                g.addEdge(node, num_tasks + 1)  # add edge from task to common destination
            else:
                if not any(node in values for values in g.graph.values()):
                    g.addEdge(node, num_tasks + 1)  # add edge from task to common destination
                if not node in g.graph.keys():
                    g.addEdge(0, node)  # add edge from common source to task
    '''
    # Now, graphs is a list of Graph instances, each representing a separate set of tasks that are connected
    for i, g in enumerate(graphs):
        print(f"Graph {i}: {g.graph}")

    return (task_mapping, g.graph)


class Data:
    def __init__(self):
        self.data_index = defaultdict(int)  # default dictionary to store data index
        self.data_size = defaultdict(int)  # default dictionary to store data size

    def addData(self, old_data_id, new_data_id, size):
        self.data_index[old_data_id] = new_data_id
        self.data_size[new_data_id] = size


class Data_req:
    def __init__(self):
        self.data_req = defaultdict(list)  # default dictionary to store data requests

    def addRequest(self, task, data):
        self.data_req[task].append(data)


def get_data_req(xml_file, task_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find the <application> element
    application = root.find('{http://opendse.sourceforge.net}application')

    # Collect dependencies and put tasks into sets
    all_data = set()
    data_objs = Data()
    data_reqs = Data_req()
    for communication in application.iter('{http://opendse.sourceforge.net}communication'):
        data_id = communication.get('id')
        size = int(communication.find(
            '{http://opendse.sourceforge.net}attributes/{http://opendse.sourceforge.net}attribute').text)
        all_data.add(data_id)
        data_objs.addData(data_id, len(all_data), size)

    for dependency in application.iter('{http://opendse.sourceforge.net}dependency'):
        source = dependency.get('source')
        destination = dependency.get('destination')

        if 'data' in source and 'data' not in destination:
            data_reqs.addRequest(destination, data_objs.data_index[source])

    # Create a mapping of old data id to new data id
    data_mapping = {old: new for old, new in data_objs.data_index.items()}
    print("Data Mapping: ", data_mapping)

    data_req_dict = defaultdict(int)

    # Print data requests
    for task, data_list in data_reqs.data_req.items():
        new_task_id = task_mapping[task]
        data_req_dict[new_task_id] = data_list
        print(f"Task {new_task_id} requests data: {data_list}")

    return data_reqs.data_req

class CoreType:
    def __init__(self):
        self.tasks = {}

    def add_task(self, new_id, old_id):
        if old_id.startswith(('t11_', 't16_', 't18_')):
            self.tasks[new_id] = 2
        else:
            self.tasks[new_id] = 1

    def print_tasks(self):
        for id, core in self.tasks.items():
            print(f'Task id: {id}, Core type: {core}')

def get_core_type(xml_file, task_mapping):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    application = root.find("{http://opendse.sourceforge.net}application")

    # Create a CoreType instance to hold tasks
    core_type = CoreType()

    # Iterate over each task
    for task in application.findall("{http://opendse.sourceforge.net}task"):
        old_id = task.get('id')

        # Add task to core_type with new id
        core_type.add_task(task_mapping[old_id], old_id)

    return core_type



def main(argv):

    tskset_file_name = 'spectestdata.xml'
    conf_file_name = 'configure_xml.json'

    try:
        opts, args = getopt.getopt(argv, "hi:j:", ["confname", "tskfname"])
    except getopt.GetoptError:
        print('tasksets_input_convertor.py -i <configuration file name> -j <the JSON task set file name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('tasksets_input_convertor.py -i <configuration file name> -j <the JSON task set file name>')
            sys.exit()
        elif opt in ("-i", "--conffname"):
            conf_file_name = str(arg)
        elif opt in ("-j", "--tskfname"):
            tskset_file_name = str(arg)

    print('Read configurations . . .')

    conf = readf.read_conf(conf_file_name)
    msets = conf['mset'][0]
    ntasks = conf['ntasks'][0]
    num_nodes_range = conf['nnodes']
    processor_a = conf['aprocessor'][0]
    processor_b = conf['bprocessor'][0]
    pc_prob = conf['pr_prob']
    sparse = conf['sparse'][0]
    util_all = conf['utilization']
    preempt_times = conf['preempt_times'][0]
    scale = conf['scale'][0]
    main_mem_time = conf['main_mem_time'][0]

    num_data_all = conf['num_data_all'][0]
    num_data_per_vertex = conf['num_data_per_vertex'][0]
    data_req_prob = conf['data_req_prob']
    num_freq_data = conf['num_freq_data'][0]
    percent_freq = conf['percent_freq'][0]
    allow_freq = conf['allow_freq'][0]

    skewness = conf['skewness'][0]
    per_heavy = conf['per_heavy'][0]
    one_type_only = conf['one_type_only'][0]

    if os.path.exists(tskset_file_name):
        print("Reading input xml file ...")
    else:
        print("Please enter correct json file name of task sets.")
        return

    task_mapping, graph_xml = get_graph(tskset_file_name)
    data_req_dict = get_data_req(tskset_file_name, task_mapping)
    core_type = get_core_type(tskset_file_name, task_mapping)
    print("Printing core types of tasks:")
    core_type.print_tasks()


    num_nodes = num_nodes_range[0] + 2

    print('Task set converting . . .')
    for ut in range(len(util_all)):
        print('Converting task set with utilization: ', util_all[ut])

        utili = float(util_all[ut] / 100)
        utilization = utili*(processor_a + processor_b)
        tasksets_name = '../experiments/inputs/tasks_pure/tasksets_pure_' + str(msets) + '_' + str(ntasks) + '_' + str(num_nodes_range) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_d' + str(num_data_per_vertex) + '_m' + str(main_mem_time) + '.npy'

        tasksets_data_name = '../experiments/inputs/tasks_data_request/tasksets_data_req_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes_range) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
            preempt_times) + '_m' + str(main_mem_time) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'

        tasksets_typed_name = '../experiments/inputs/tasks_typed/tasksets_typed_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes_range) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
            preempt_times) + '_m' + str(main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
            one_type_only) + '.npy'

        tasksets = []
        tasksets_typed = []
        tasksets_data = []
        if msets > 1:
            print("Please note, this script can only handle one set at a time!")
            return

        msets = 1

        for i in range(msets):
            taskset = []
            taskset_typed = []
            taskset_data = []

            # generate the lower bound of the wcet of each task set
            util_lower_bound_set = []
            # generate the number of nodes of each task
            num_nodes_all = []
            # generate the period of each task
            periods = []
            for tsk in range(ntasks):
                periods.append(1)
                num_nodes_all.append(num_nodes)
                # worst case data accessing time = main_mem_time * (preempt_times + 1)
                lower_bound_tsk_temp = num_nodes * num_data_per_vertex * main_mem_time * (
                            preempt_times + 1) / scale
                util_lower_bound_set.append(lower_bound_tsk_temp)

            if sum(util_lower_bound_set) >= utilization:
                print("Too many preempt times, unable to generate the task set")
                return

            util_upper_bound_set = [8] * ntasks
            # drs(number, sum, upper_bound, lowe_bound)
            util_tasks = drs(ntasks, utilization, util_upper_bound_set, util_lower_bound_set)

            j = 0
            while j < ntasks:



                g_temp = gen.Graph(num_nodes)
                period = int(1)
                g_temp.period = period * scale
                # implicit deadline
                g_temp.deadline = period * scale

                # generate the lower bound of the node of each task
                util_lower_bound_tsk = []
                util_upper_bound_tsk = [1] * (num_nodes - 2)
                for n in range(1, num_nodes - 1):
                    if len(data_req_dict[n]) > 0:
                        num_data_vertex = len(data_req_dict[n])
                    else:
                        num_data_vertex = 1
                    util_lower_bound_tsk.append(
                        main_mem_time * num_data_vertex * (preempt_times + 1) / period / scale)

                # common source node and one common end node have 0 utilization
                util_nodes = drs(num_nodes - 2, util_tasks[j], util_upper_bound_tsk, util_lower_bound_tsk)
                util_nodes.insert(0, 0)
                util_nodes.append(0)

                g_temp.utilization = util_tasks[j]
                g_temp.utilizations = util_nodes

                WCETs_float = [nd_u * period * scale for nd_u in util_nodes]
                WCETs_int = list(map(int, WCETs_float))

                # print(WCETs_int)
                g_temp.weights = WCETs_int
                g_temp.priorities = [0] * num_nodes

                # define the graph structure
                struct = defaultdict(list)

                for source in range(0, num_nodes - 1):
                    struct[source] = copy.deepcopy(graph_xml[source])

                print("number of nodes: ", num_nodes)
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
                        struct[node].append(num_nodes - 1)
                    if len(struct[node]) < 1:
                        print('Length3 : ', len(struct[node]))
                        print('something wrong 0!!!')

                # append the last item
                struct[num_nodes - 1]

                g_temp.graph = copy.deepcopy(dict(sorted(struct.items())))

                for node in range(1, num_nodes - 1):
                    if len(g_temp.graph[node]) < 1:
                        print('something wrong!!!')

                g_temp.predecessors = gen.find_predecessors(g_temp.graph, num_nodes)

                # check if the longest path is longer than the period already
                # the total execution time is no longer than the period
                longest_path = gen.find_longest_path(g_temp.graph, g_temp.weights)[1]
                if sum(g_temp.weights) <= g_temp.period or longest_path <= g_temp.period:
                    g_temp.cp = int(longest_path)
                    g_temp.deadlines = gen.calculate_deadlines(g_temp.graph, g_temp.weights, g_temp.deadline)
                    g_temp.tsk_id = int(j)
                    taskset.append(g_temp)

                    type_temp = typed.Type(num_nodes)

                    type_temp.typed[0] = 1
                    for tp in range(1, num_nodes - 1):
                        type_temp.typed[tp] = core_type.tasks[tp]

                    type_temp.typed[num_nodes - 1] = 1

                    # handle the typed infomation
                    # temp weights for vertices only in processor A
                    temp_weights_a = copy.deepcopy(g_temp.weights)
                    # temp weights for vertices only in processor B
                    temp_weights_b = copy.deepcopy(g_temp.weights)

                    for node in range(0, num_nodes):
                        if type_temp.typed[node] == 1:
                            temp_weights_b[node] = 0

                        else:
                            temp_weights_a[node] = 0

                    # In the following some additional information is calculated for future usage
                    # calculate the L_i^a and L_i^b
                    La = gen.find_longest_path(g_temp.graph, temp_weights_a)[1]
                    Lb = gen.find_longest_path(g_temp.graph, temp_weights_b)[1]

                    if La == g_temp.period / 3 or La == g_temp.period / 2:
                        La += 1
                    if Lb == g_temp.period / 3 or Lb == g_temp.period / 2:
                        Lb += 1

                    type_temp.cpA = La
                    type_temp.cpB = Lb

                    # Accumulative execution time of vertices on type A and B
                    type_temp.utilizationA = sum(temp_weights_a) / g_temp.period
                    type_temp.utilizationB = sum(temp_weights_b) / g_temp.period

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

                    # data
                    requested_data = data_req.ReqData(num_nodes)

                    requested_data.req_data[0].append(hex(0))
                    requested_data.req_prob[0].append(1)
                    for n in range(1, num_nodes - 1):
                        if len(data_req_dict[n]) > 0:
                            for d in range(len(data_req_dict[n])):
                                requested_data.req_data[n].append(hex(data_req_dict[n][d]))
                                requested_data.req_prob[n].append(1)
                        else:
                            requested_data.req_data[n].append(hex(0))
                            requested_data.req_prob[n].append(1)

                    requested_data.req_data[num_nodes - 1].append(hex(0))
                    requested_data.req_prob[num_nodes - 1].append(1)

                    taskset_data.append(requested_data)

                    print("graph: ", g_temp.graph)
                    print("weight: ", g_temp.weights)
                    print("deadline: ", g_temp.deadlines)
                    print("typed: ", type_temp.typed)
                    print("data: ", requested_data.req_data)

                    j = j + 1

        taskset_temp = copy.deepcopy(taskset)
        sorted_index = [i[0] for i in sorted(enumerate(taskset_temp), key=lambda x: (x[1].period, -x[1].utilization))]
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























