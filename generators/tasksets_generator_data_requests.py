from __future__ import division
import numpy as np
import data_requests as data_req
import os
import math
import sys
import getopt
import read_configuration as readf


def main(argv):

    # default json configuration file name
    conf_file_name = 'configure.json'

    try:
        opts, args = getopt.getopt(argv, "hi:", ["conf_file_name"])
    except getopt.GetoptError:
        print('read_configuration.py -i <the JSON configuration file name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('read_configuration.py -i <the JSON configuration file name>')
            sys.exit()
        elif opt in ("-i", "--conffname"):
            conf_file_name = str(arg)

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

    for ut in range(len(util_all)):

        print("Reading the original task set ...")
        utili = float(util_all[ut] / 100)
        tasksets_pure_name = '../experiments/inputs/tasks_pure/tasksets_pure_' + str(msets) + '_' + str(ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(main_mem_time) + '.npy'
        tasksets_pure = np.load(tasksets_pure_name, allow_pickle=True)

        print('Generating data requests for task sets with utilization: ', util_all[ut])
        tasksets_data = data_req.generated_requested_data(msets, tasksets_pure, num_data_all, num_freq_data, percent_freq, allow_freq)
        tasksets_data_name = '../experiments/inputs/tasks_data_request/tasksets_data_req_' + str(msets) + '_' + str(ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(main_mem_time) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(allow_freq) + '.npy'
        np.save(tasksets_data_name, np.array(tasksets_data, dtype=object))


if __name__ == "__main__":
    main(sys.argv[1:])


