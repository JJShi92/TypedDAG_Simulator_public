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


# The dictionary of requested data of each task
# Each vertex can request multiple addresses of data
class ReqData:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.req_data = defaultdict(list)  # the requested data for each vertex
        self.req_prob = defaultdict(list)  # the probability of requesting the specific data


# Generate the requested data of each task set
# num_data_all: the number of all the available data
# num_data_per_vertex: the maximum of data that each vertex can request
# num_freq_data: number of the frequently requested data
# percent_freq: the percentage of requesting the frequently requested data
# data_req_prob: the probabilities that each data address is requested
# mod0: fully randomly generate the requested data
# mod1: control the percent of frequently requested data
def generated_requested_data(msets, task_sets_org, num_data_all, num_data_per_vertex, num_freq_data, percent_freq, data_req_prob, mod):
    data_sets = []
    for m in range(msets):
        data_set = []
        for tsk in range(len(task_sets_org[m])):
            requested_data = ReqData(task_sets_org[m][tsk].V)
            # fully random generation
            if mod == 0:
                for v in range(1, task_sets_org[m][tsk].V-1):
                    req_data_tsk = random.sample(range(1, num_data_all+1), num_data_per_vertex)
                    for d in range(num_data_per_vertex):
                        requested_data.req_data[v].append(hex(req_data_tsk[d]))

            # control the frequently requested data
            else:
                for v in range(1, task_sets_org[m][tsk].V-1):
                    num_freq = round(num_data_per_vertex * percent_freq)
                    req_data_tsk_freq = random.sample(range(1, num_freq_data+1), num_freq)
                    req_data_tsk_nfreq = random.sample(range(num_freq_data+1, num_data_all+1), num_data_per_vertex - num_freq)
                    for d in range(num_data_per_vertex):
                        # request the frequently requested data
                        if d < num_freq:
                            requested_data.req_data[v].append(hex(req_data_tsk_freq[d]))
                        else:
                            # request the normal data
                            requested_data.req_data[v].append(hex(req_data_tsk_nfreq[d-num_freq]))

            for d in range(num_data_per_vertex):
                # make sure the common source node and common end node do not request any data
                requested_data.req_data[0].append(hex(0))
                requested_data.req_data[task_sets_org[m][tsk].V-1].append(hex(0))

            # append the data request probability for each vertex
            for v in range(task_sets_org[m][tsk].V):
                requested_data.req_prob[v] = data_req_prob

            data_set.append(requested_data)

        data_sets.append(data_set)

    return data_sets
