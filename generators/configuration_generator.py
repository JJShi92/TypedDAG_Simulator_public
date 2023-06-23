# Configuration generator
# Define a uniform configuration file that can be used for all scripts

import csv
import json

# All the supported configurations are as follows:
# msets: number of sets
# ntasks: number of tasks for each set
# If the given ntasks < 1, the number of tasks are generated randomly due to the sparse parameter
# spars-0: [0.5 * max(aprocessor, bprocessor), 2 * max(aprocessor, bprocessor)]
# spars-1: [(aprocessor + aprocessor), 2 * (aprocessor + aprocessor)]
# spars-2: [0.25 * (aprocessor + aprocessor), (aprocessor + aprocessor)]
# aprocessor: number of processor A
# bprocessor: number of processor B
# pc_prob: the lower bound and upper bound of probability of two vertices have edge
# The real probability \in [pc_prob_l, pc_prob_h]
# utilization: total utilization for a set of tasks
# scale: the scale to keep all the parameters are integers

# skewness: controls the skewness of the skewed tasks
# e.g., 10% nodes for the task is assigned on A core, and others on B core (heavy^b task)
# per_heavy: the percentage of heavy^a or heavy^b tasks, e.g., 0%, 25%, 50%, 75%, and 100%
# one_type_only: if allow a task only require one type of processor:
# i.e., 0: not allowed; 1: allowed, the percentage can be defined by mod_2 (if needed)

# num_data_all: the number of all the available data
# num_data_per_vertex: the maximum of data that each vertex can request
# num_freq_data: number of the frequently requested data
# percent_freq: the percentage of requesting the frequently requested data
# allow_freq-0: fully randomly generate the requested data regardless of the frequently requested data
# allow_freq-1: control the percent of frequently requested data
# data_req_prob: the probability of requesting the specific data

# main_mem_size: the size for main memory, assume a very large number can store all the requested data
# main_mem_time: the time for data access from main memory
# fast_mem_size: the size for fast memory
# fast_mem_time: time time for data access from fast memory
# l1_cache_size: the size for l1 cache
# l1_cache_time: the time for data access from l1 cache

# try_avg_case: when the wcet cannot pass the schedulability test, do we try average case execution time (acet)
# avg_ratio: the ratio of acet/wcet
# std_dev: the the standard deviation for generating real case execution time when simulating the schedule
# tolerate_pa: the upper bound of tolerable number of type A processor when the current number of type A processor is not enough
# tolerate_pb: the upper bound of tolerable number of type B processor when the current number of type B processor is not enough
# rho_greedy: the setting of rho for greedy federated schedule algorithm, 0 < rho <= 0.5
# rho_imp_fed: the setting of rho for our improved federated schedule algorithm, rho = 1/7.25

# Define the configure setting names and values
conf_data = {
    'mset': [10],
    'ntasks': [0],
    'nnodes': [50, 100],
    'scale': [10**6],
    'aprocessor': [16],
    'bprocessor': [4],
    'pr_prob': [0.2, 0.7],
    'utilization': [30, 40],
    'preempt_times': [4],
    'sparse': [0],
    'skewness': [0],
    'per_heavy': [2],
    'one_type_only': [1],
    'num_data_all': [100],
    'num_data_per_vertex': [5],
    'num_freq_data': [20],
    'percent_freq': [0.1],
    'allow_freq': [0],
    'data_req_prob': [1, 0.8, 0.6, 0.3, 0.1],
    'main_mem_size': [10000],
    'main_mem_time': [100],
    'fast_mem_size': [1000],
    'fast_mem_time': [10],
    'l1_cache_size': [100],
    'l1_cache_time': [1],
    'try_avg_case': [True],
    'avg_ratio': [0.5],
    'min_ratio': [0.1],
    'std_dev': [0.3],
    'tolerate_pa': [20],
    'tolerate_pb': [12],
    'rho_greedy': [0.5],
    'rho_imp_fed': [1/7.25]
}

# conf_names = ['mset', 'ntasks', 'aprocessor', 'bprocesspr', 'pr_prob_l', 'pc_prob_h', 'utilization', 'sparse', 'skewness', 'per_heavy', 'one_type_only', 'num_data_all', 'num_freq_data', 'percent_freq', 'allow_freq']

json_file_name = 'configure.json'

print("Generating configuration file ... ")

with open(json_file_name, 'w') as json_file:
    json.dump(conf_data, json_file, indent=4)

print(f"Configuration file '{json_file_name}' has been created successfully.")