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
class ReqData:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.req_data = defaultdict(hex)  # the requested data for each vertex


# Generate the requested data of each task set
# num_data_all: the number of all the available data
# num_freq_data: number of the frequently requested data
# percent_freq: the percentage of requesting the frequently requested data
# mod0: fully randomly generate the requested data
# mod1: control the percent of frequently requested data
def generated_requested_data(msets, task_sets_org, num_data_all, num_freq_data, percent_freq, mod):
    data_sets = []
    for m in range(msets):
        data_set = []
        for tsk in range(len(task_sets_org[m])):
            requested_data = ReqData(task_sets_org[m][tsk].V)
            # fully random generation
            if mod == 0:
                for v in range(task_sets_org[m][tsk].V):
                    requested_data.req_data[v] = hex(random.randint(1, num_data_all))

            # control the frequently requested data
            else:
                for v in range(task_sets_org[m][tsk].V):
                    # request the frequently requested data
                    if random.uniform(0, 1) < percent_freq:
                        requested_data.req_data[v] = hex(random.randint(1, num_freq_data))
                    else:
                        # request the normal data
                        requested_data.req_data[v] = hex(random.randint(num_freq_data+1, num_data_all))

            # make sure the common source node and common end node do not request any data
            requested_data.req_data[0] = hex(0)
            requested_data.req_data[task_sets_org[m][tsk].V-1] = hex(0)

            data_set.append(requested_data)

        data_sets.append(data_set)

    return data_sets
