# The script to generate the schedule of given configurations

from __future__ import division
import numpy as np
import os
import math
import sys
import getopt
sys.path.append('../')
from algorithms import sched_sim
from generators import data_requests
from generators import generator_pure_dict
from generators import typed_core_allocation
from generators import read_configuration as readf
from algorithms import affinity_han as han
from algorithms import affinity_improved_fed as imp_fed
from algorithms import misc
from algorithms import affinity_raw as raw
from algorithms import sched_sim as sched_sim

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

    skewness = conf['skewness'][0]
    per_heavy = conf['per_heavy'][0]
    one_type_only = conf['one_type_only'][0]

    num_data_all = conf['num_data_all'][0]
    num_freq_data = conf['num_freq_data'][0]
    percent_freq = conf['percent_freq'][0]
    allow_freq = conf['allow_freq'][0]
    main_mem_size = conf['main_mem_size'][0]
    main_mem_time = conf['main_mem_time'][0]
    fast_mem_size = conf['fast_mem_size'][0]
    fast_mem_time = conf['fast_mem_time'][0]
    l1_cache_size = conf['l1_cache_size'][0]
    l1_cache_time = conf['l1_cache_time'][0]
    try_avg_case = conf['try_avg_case'][0]
    avg_ratio = conf['avg_ratio'][0]
    min_ratio = conf['min_ratio'][0]
    std_dev = conf['std_dev'][0]
    tolerate_pa = conf['tolerate_pa'][0]
    tolerate_pb = conf['tolerate_pb'][0]
    rho_greedy = conf['rho_greedy'][0]
    rho_imp_fed = conf['rho_imp_fed'][0]

    util_all = [30]

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


        for s in range(msets):
            print("Generate the schedule for set: ", s)

            print("Try to find the original affinity allocation with WCET ...")

            aff_wcet_name = '../experiments/outputs/affinity_allocation/aff_wcet_' + str(msets) + '_' + str(
                    ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                    pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
                    preempt_times) + '_m' + str(
                    main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                    one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                    percent_freq) + '_' + str(
                    allow_freq) + '.npy'
            if os.path.exists(aff_wcet_name):
                affinities, processors = np.load(aff_wcet_name, allow_pickle=True)


                schedule_wcet = sched_sim.typed_dag_schedule_gen(tasksets_pure[s], tasksets_typed[s], tasksets_data[s], affinities, processors[0], processors[1], max_time, scale,
                       preempt_times, main_mem_size, main_mem_time, fast_mem_size, fast_mem_time, l1_cache_size,
                       l1_cache_time, avg_ratio, min_ratio, std_dev)

                sched_wcet_name = '../experiments/outputs/schedule/sched_wcet_' + str(msets) + '_' + str(
                    ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                    pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                    int(math.log10(scale))) + '_' + str(
                    preempt_times) + '_m' + str(
                    main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                    one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                    percent_freq) + '_' + str(
                    allow_freq) + '.npy'

                np.save(sched_wcet_name, np.array(schedule_wcet, dtype=object))
            else:
                print("The original affinity allocation with WCET does not exist, try ACET case")

                aff_acet_name = '../experiments/outputs/affinity_allocation/aff_acet_' + str(msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                        percent_freq) + '_' + str(
                        allow_freq) + '.npy'
                if os.path.exists(aff_acet_name):
                    affinities, processors = np.load(aff_acet_name, allow_pickle=True)

                    schedule_acet = sched_sim.typed_dag_schedule_gen(tasksets_pure[s], tasksets_typed[s], tasksets_data[s],
                                                           affinities, processors[0], processors[1], max_time, scale,
                                                           preempt_times, main_mem_size, main_mem_time, fast_mem_size,
                                                           fast_mem_time, l1_cache_size,
                                                           l1_cache_time, avg_ratio, min_ratio, std_dev)

                    sched_acet_name = '../experiments/outputs/schedule/sched_acet_' + str(msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                        percent_freq) + '_' + str(
                        allow_freq) + '.npy'

                    np.save(sched_acet_name, np.array(schedule_acet, dtype=object))
                else:
                    print("The affinity allocation with ACET does not exist")

                print("Try the affinity allocation with maximum number of required processor A and processor B")

                aff_tol_wcet_name = '../experiments/outputs/affinity_allocation/aff_tolerate_wcet_' + str(msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + str(s) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                        percent_freq) + '_' + str(
                        allow_freq) + '.npy'
                if os.path.exists(aff_tol_wcet_name):
                    affinities, processors = np.load(aff_tol_wcet_name, allow_pickle=True)
                    schedule_tol_wcet = sched_sim.typed_dag_schedule_gen(tasksets_pure[s], tasksets_typed[s], tasksets_data[s],
                                                           affinities, processors[0], processors[1], max_time, scale,
                                                           preempt_times, main_mem_size, main_mem_time, fast_mem_size,
                                                           fast_mem_time, l1_cache_size,
                                                           l1_cache_time, avg_ratio, min_ratio, std_dev)

                    sched_tol_wcet_name = '../experiments/outputs/schedule/sched_tol_wcet_' + str(
                        msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                        percent_freq) + '_' + str(
                        allow_freq) + '.npy'

                    np.save(sched_tol_wcet_name, np.array(schedule_tol_wcet, dtype=object))
                else:
                    print("The affinity allocation with maximum number of required processor A and processor B does not exist")

                if not os.path.exists(aff_acet_name) and not os.path.exists(aff_tol_wcet_name):
                    print(
                        "Try average case execution time with tolerate A and B processors instead of worst case execution time ...")

                    aff_tol_acet_name = '../experiments/outputs/affinity_allocation/aff_tolerate_acet_' + str(
                                msets) + '_' + str(
                                ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                                processor_b) + '_q' + str(
                                pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                                int(math.log10(scale))) + '_' + str(
                                preempt_times) + '_m' + str(
                                main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                                one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                                percent_freq) + '_' + str(
                                allow_freq) + '.npy'
                    if os.path.exists(aff_tol_acet_name):
                        affinities, processors = np.load(aff_tol_acet_name, allow_pickle=True)
                        schedule_tol_acet = sched_sim.typed_dag_schedule_gen(tasksets_pure[s], tasksets_typed[s],
                                                                   tasksets_data[s],
                                                                   affinities, processors[0], processors[1], max_time,
                                                                   scale,
                                                                   preempt_times, main_mem_size, main_mem_time,
                                                                   fast_mem_size,
                                                                   fast_mem_time, l1_cache_size,
                                                                   l1_cache_time, avg_ratio, min_ratio, std_dev)

                        sched_tol_acet_name = '../experiments/outputs/schedule/sched_tol_acet_' + str(
                            msets) + '_' + str(
                            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                            processor_b) + '_q' + str(
                            pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                            int(math.log10(scale))) + '_' + str(
                            preempt_times) + '_m' + str(
                            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                            one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                            percent_freq) + '_' + str(
                            allow_freq) + '.npy'
                        np.save(sched_tol_acet_name, np.array(schedule_tol_acet, dtype=object))

                    else:
                        aff_raw_name = '../experiments/outputs/affinity_allocation/aff_raw_wcet_' + str(
                            msets) + '_' + str(
                            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                            processor_b) + '_q' + str(
                            pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                            int(math.log10(scale))) + '_' + str(
                            preempt_times) + '_m' + str(
                            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                            one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                            percent_freq) + '_' + str(
                            allow_freq) + '.npy'
                        if os.path.exists(aff_raw_name):
                            print("Only have the raw affinities by using type aware global schedule...")
                            affinities, processors = np.load(aff_raw_name, allow_pickle=True)
                            schedule_raw = sched_sim.typed_dag_schedule_gen(tasksets_pure[s], tasksets_typed[s],
                                                                       tasksets_data[s],
                                                                       affinities, processors[0], processors[1],
                                                                       max_time,
                                                                       scale,
                                                                       preempt_times, main_mem_size, main_mem_time,
                                                                       fast_mem_size,
                                                                       fast_mem_time, l1_cache_size,
                                                                       l1_cache_time, avg_ratio, min_ratio, std_dev)

                            sched_raw_name = '../experiments/outputs/schedule/sched_raw_' + str(
                                msets) + '_' + str(
                                ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                                processor_b) + '_q' + str(
                                pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                                int(math.log10(scale))) + '_' + str(
                                preempt_times) + '_m' + str(
                                main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                                one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                                percent_freq) + '_' + str(
                                allow_freq) + '.npy'
                            np.save(sched_raw_name, np.array(schedule_raw, dtype=object))


                        else:
                            print("There is no feasible affinity files, please generate affinity at first")

if __name__ == "__main__":
    main(sys.argv[1:])