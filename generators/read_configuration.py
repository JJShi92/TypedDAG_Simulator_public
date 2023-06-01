from __future__ import division
import os
import math
import sys
import getopt
import json

class Conf_settings:
    def __init__(self):
        self.msets = int
        self.ntasks = int
        self.nnodes = List[int]
        self.scale = int
        self.processor_a = int
        self.processor_b = int
        self.pc_prob = List[float]
        self.utilization = List[int]
        self.preempt_times = int
        self.sparse = int
        self.skewness = int
        self.per_heavy = int
        self.one_type_only = int
        self.num_data_all = int
        self.num_freq_data = int
        self.percent_freq = float
        self.allow_freq = int
        self.main_mem_size = int
        self.main_mem_time = int
        self.fast_mem_size = int
        self.fast_mem_time = int
        self.l1_cache_size = int
        self.l1_cache_time = int


def read_conf(conf_file_name):

    with open(conf_file_name, 'r') as json_file:
        data = json.load(json_file)

    conf_names = list(data.keys())

    # Initialize a dictionary to store attribute settings
    conf_settings = {name: set() for name in conf_names}

    # Extract the settings for each conf
    for name in conf_names:
        values = data[name]
        conf_settings[name].update(values)

    # Sort the values in each setting
    conf_settings = {key: sorted(values) for key, values in conf_settings.items()}

    return conf_settings
