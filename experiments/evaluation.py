# The script to generate the schedule of given configurations

from __future__ import division
import numpy as np
import os
import math
import sys
import getopt
sys.path.append('../')
from algorithms import sched_sim_test as sched_sim
from generators import data_requests
from generators import generator_pure_dict
from generators import typed_core_allocation
from generators import read_configuration as readf


def main(argv):

    # default json configuration file name
    conf_file_name = '../generators/configure.json'
    aff_mod = 0
    max_time = 10 ** 9

    try:
        opts, args = getopt.getopt(argv, "hi:m:", ["conf_file_name", "aff_mod"])
    except getopt.GetoptError:
        print('read_configuration.py -i <the JSON configuration file name> -m <affinity definition mod>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('read_configuration.py -i <the JSON configuration file name> -m <affinity definition mod>')
            sys.exit()
        elif opt in ("-o", "--conffname"):
            conf_file_name = str(arg)
        elif opt in ("-m", "--affmod"):
            aff_mod = int(arg)

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

    skewness = conf['skewness'][0]
    per_heavy = conf['per_heavy'][0]
    one_type_only = conf['one_type_only'][0]

    num_data_all = conf['num_data_all'][0]
    num_freq_data = conf['num_freq_data'][0]
    percent_freq = conf['percent_freq'][0]
    allow_freq = conf['allow_freq'][0]

    if aff_mod == 0:
        aff_approach = 'han'

    util_all = [20]
    for ut in range(len(util_all)):
    #for ut in range(1):
        print("Reading the original task set ...")
        utili = float(util_all[ut] / 100)
        # original task set
        tasksets_pure_name = '../experiments/inputs/tasks_pure/tasksets_pure_' + str(msets) + '_' + str(ntasks) + '_' + str(
            num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(
            utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(main_mem_time) + '.npy'
        tasksets_pure = np.load(tasksets_pure_name, allow_pickle=True)
        # data requests
        tasksets_data_name = '../experiments/inputs/tasks_data_request/tasksets_data_req_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(
            main_mem_time) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(
            allow_freq) + '.npy'
        tasksets_data = np.load(tasksets_data_name, allow_pickle=True)
        # typed information
        tasksets_typed_name = '../experiments/inputs/tasks_typed/tasksets_typed_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(
            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(one_type_only) + '.npy'
        tasksets_typed = np.load(tasksets_typed_name, allow_pickle=True)


        sched_sim_results_name = '../experiments/outputs/' + aff_approach + '/sched_sim_' + str(msets) + '_' + str(
            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
            pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(
            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(
            allow_freq) + '.npy'

        sched_sim_results = sched_sim.typed_dag_schedule_gen(tasksets_pure, tasksets_typed, tasksets_data, processor_a, processor_b, max_time, scale, preempt_times, 0)

        np.save(sched_sim_results_name, sched_sim_results)

if __name__ == "__main__":
    main(sys.argv[1:])