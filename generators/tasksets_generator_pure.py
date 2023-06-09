from __future__ import division
import numpy as np
import os
import math
import sys
import getopt
import generator_pure_dict as gen
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

    print('Task set generating . . .')
    for ut in range(len(util_all)):
        print('Generating task set with utilization: ', util_all[ut])

        utili = float(util_all[ut] / 100)
        utilization = utili*(processor_a + processor_b)
        tasksets_name = '../experiments/inputs/tasks_pure/tasksets_pure_' + str(msets) + '_' + str(ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(pc_prob) + '_u' + str(utili) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(preempt_times) + '_m' + str(main_mem_time) + '.npy'
        tasksets = gen.generate_tsk_dict(msets, ntasks, num_nodes, processor_a, processor_b, pc_prob, utilization, sparse, scale, preempt_times, main_mem_time)

        np.save(tasksets_name, np.array(tasksets, dtype=object))


if __name__ == "__main__":
    main(sys.argv[1:])
