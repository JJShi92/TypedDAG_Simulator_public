# The script to generate the affinities for a set of tasks of given configurations
from __future__ import division
import numpy as np
import os
import math
from collections import defaultdict
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

    avg_or_tolerate = True

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
    std_dev = conf['std_dev'][0]
    tolerate_pa = conf['tolerate_pa'][0]
    tolerate_pb = conf['tolerate_pb'][0]
    rho_greedy = conf['rho_greedy'][0]
    rho_imp_fed = conf['rho_imp_fed'][0]

    if aff_mod == 0:
        aff_approach = 'improved_fed'

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
            print("Finding affinities ...")

            # First try these approaches with original WCET
            print("Try Han's approach with EMU partition approach ...")
            aff_han = han.sched_han(tasksets_pure[s], tasksets_typed[s], processor_a, processor_b, 1)
            if aff_han[0]:
                affinities_han = defaultdict(int, sorted(aff_han[1].items(), key=lambda x: x[0]))
                max_a_han, max_b_han = misc.find_max_ab_index(affinities_han)
                unused_a_cores_han = processor_a - max_a_han - 1
                if unused_a_cores_han > 0:
                    affinities_han = misc.adjust_unused_cores(affinities_han, unused_a_cores_han)
                used_a_han = max_a_han + 1
                used_b_han = max_b_han - processor_a + 1
                print("Successfully pass the schedulability test, require core number: ", used_a_han, used_b_han)

            print("Try improved federated scheduling approach ...")
            aff_imp = imp_fed.improved_federated_p3(tasksets_pure[s], tasksets_typed[s], processor_a, processor_b, rho_imp_fed)
            if aff_imp[0]:
                affinities_imp = defaultdict(int, sorted(aff_imp[1].items(), key=lambda x: x[0]))
                max_a_imp, max_b_imp = misc.find_max_ab_index(affinities_imp)
                unused_a_cores_imp = processor_a - max_a_imp - 1
                if unused_a_cores_imp > 0:
                    affinities_imp = misc.adjust_unused_cores(affinities_imp, unused_a_cores_imp)
                used_a_imp = max_a_imp + 1
                used_b_imp = max_b_imp - processor_a + 1
                print("Successfully pass the schedulability test, require core number: ", used_a_imp, used_b_imp)

            if aff_han[0] or aff_imp[0]:
                avg_or_tolerate = False
                affinity = []
                if aff_han[0] and not aff_imp[0]:
                    affinity.append(affinities_han)
                    affinity.append([used_a_han, used_b_han])
                elif not aff_han[0] and aff_imp[0]:
                    affinity.append(affinities_imp)
                    affinity.append([used_a_imp, used_b_imp])
                else:
                    if used_a_han + used_b_han < used_a_imp + used_b_imp:
                        affinity.append(affinities_han)
                        affinity.append([used_a_han, used_b_han])
                    else:
                        affinity.append(affinities_imp)
                        affinity.append([used_a_imp, used_b_imp])

                aff_name = '../experiments/outputs/affinity_allocation/aff_wcet_' + str(msets) + '_' + str(
                    ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                    pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(int(math.log10(scale))) + '_' + str(
                    preempt_times) + '_m' + str(
                    main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                    one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                    percent_freq) + '_' + str(
                    allow_freq) + '.npy'
                np.save(aff_name, np.array(affinity, dtype=object))
                print("Optimized affinity information has been saved.")

            # If both approaches with original WCET are infeasible
            #if not aff_han[0] and not aff_imp[0]:
            else:
                print("Two approaches are both infeasible")
                # Try average case execution time
                if try_avg_case:
                    print("Try average case execution time instead of worst case execution time ...")

                    print("Try Han's approach with EMU partition approach ...")
                    aff_han = han.sched_han(misc.average_case_convertor_taskset(tasksets_pure[s], avg_ratio), misc.average_case_convertor_typed(tasksets_typed[s], avg_ratio), processor_a, processor_b, 1)
                    if aff_han[0]:
                        affinities_han = defaultdict(int, sorted(aff_han[1].items(), key=lambda x: x[0]))
                        max_a_han, max_b_han = misc.find_max_ab_index(affinities_han)
                        unused_a_cores_han = processor_a - max_a_han - 1
                        if unused_a_cores_han > 0:
                            affinities_han = misc.adjust_unused_cores(affinities_han, unused_a_cores_han)
                        used_a_han = max_a_han + 1
                        used_b_han = max_b_han - processor_a + 1
                        print("Successfully pass the schedulability test, require core number: ", used_a_han,
                              used_b_han)

                    print("Try improved federated scheduling approach ...")
                    aff_imp = imp_fed.improved_federated_p3(misc.average_case_convertor_taskset(tasksets_pure[s], avg_ratio), misc.average_case_convertor_typed(tasksets_typed[s], avg_ratio), processor_a,
                                                            processor_b, rho_imp_fed)
                    if aff_imp[0]:
                        affinities_imp = defaultdict(int, sorted(aff_imp[1].items(), key=lambda x: x[0]))
                        max_a_imp, max_b_imp = misc.find_max_ab_index(affinities_imp)
                        unused_a_cores_imp = processor_a - max_a_imp - 1
                        if unused_a_cores_imp > 0:
                            affinities_imp = misc.adjust_unused_cores(affinities_imp, unused_a_cores_imp)
                        used_a_imp = max_a_imp + 1
                        used_b_imp = max_b_imp - processor_a + 1
                        print("Successfully pass the schedulability test, require core number: ", used_a_imp,
                              used_b_imp)

                    if aff_han[0] or aff_imp[0]:
                        avg_or_tolerate = False
                        affinity = []
                        if aff_han[0] and not aff_imp[0]:
                            affinity.append(affinities_han)
                            affinity.append([used_a_han, used_b_han])
                        elif not aff_han[0] and aff_imp[0]:
                            affinity.append(affinities_imp)
                            affinity.append([used_a_imp, used_b_imp])
                        else:
                            if used_a_han + used_b_han <= used_a_imp + used_b_imp:
                                affinity.append(affinities_han)
                                affinity.append([used_a_han, used_b_han])
                            else:
                                affinity.append(affinities_imp)
                                affinity.append([used_a_imp, used_b_imp])

                        aff_name = '../experiments/outputs/affinity_allocation/aff_acet_' + str(msets) + '_' + str(
                            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                            pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                            int(math.log10(scale))) + '_' + str(
                            preempt_times) + '_m' + str(
                            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                            one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                            percent_freq) + '_' + str(
                            allow_freq) + '.npy'
                        np.save(aff_name, np.array(affinity, dtype=object))
                        print("Optimized affinity information with acet has been saved.")


                # Try to find the maximum number of required processor A and processor B
                print("Try to find the maximum number of required processor A and processor B ...")

                print("Try Han's approach with EMU partition approach and tolerate A and B processor ...")
                aff_han = han.sched_han(tasksets_pure[s], tasksets_typed[s], tolerate_pa, tolerate_pb, 1)
                if aff_han[0]:
                    affinities_han = defaultdict(int, sorted(aff_han[1].items(), key=lambda x: x[0]))
                    max_a_han, max_b_han = misc.find_max_ab_index(affinities_han)
                    unused_a_cores_han = processor_a - max_a_han - 1
                    if unused_a_cores_han > 0:
                        affinities_han = misc.adjust_unused_cores(affinities_han, unused_a_cores_han)
                    used_a_han = max_a_han + 1
                    used_b_han = max_b_han - processor_a + 1
                    print("Successfully pass the schedulability test, require core number: ", used_a_han, used_b_han)

                print("Try improved federated scheduling approach with tolerate A and B processor ...")
                aff_imp = imp_fed.improved_federated_p3(tasksets_pure[s], tasksets_typed[s], tolerate_pa,
                                                        tolerate_pb, rho_imp_fed)
                if aff_imp[0]:
                    affinities_imp = defaultdict(int, sorted(aff_imp[1].items(), key=lambda x: x[0]))
                    max_a_imp, max_b_imp = misc.find_max_ab_index(affinities_imp)
                    unused_a_cores_imp = processor_a - max_a_imp - 1
                    if unused_a_cores_imp > 0:
                        affinities_imp = misc.adjust_unused_cores(affinities_imp, unused_a_cores_imp)
                    used_a_imp = max_a_imp + 1
                    used_b_imp = max_b_imp - processor_a + 1
                    print("Successfully pass the schedulability test, require core number: ", used_a_imp, used_b_imp)

                if aff_han[0] or aff_imp[0]:
                    avg_or_tolerate = False
                    affinity = []
                    if aff_han[0] and not aff_imp[0]:
                        affinity.append(affinities_han)
                        affinity.append([used_a_han, used_b_han])
                    elif not aff_han[0] and aff_imp[0]:
                        affinity.append(affinities_imp)
                        affinity.append([used_a_imp, used_b_imp])
                    else:
                        if used_a_han + used_b_han <= used_a_imp + used_b_imp:
                            affinity.append(affinities_han)
                            affinity.append([used_a_han, used_b_han])
                        else:
                            affinity.append(affinities_imp)
                            affinity.append([used_a_imp, used_b_imp])

                    aff_name = '../experiments/outputs/affinity_allocation/aff_tolerate_wcet_' + str(msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(s) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_freq_data) + '_' + str(
                        percent_freq) + '_' + str(
                        allow_freq) + '.npy'
                    np.save(aff_name, np.array(affinity, dtype=object))
                    print("Optimized affinity information with tolerate A and B processors has been saved.")


                # If both approaches with original WCET and maximum tolerate A and B processors are still infeasible
                if avg_or_tolerate:
                    print("Two approaches with maximum tolerate A and B processors are still both infeasible")
                    # Try average case execution time with tolerate A and B processors
                    if try_avg_case:
                        print("Try average case execution time with tolerate A and B processors instead of worst case execution time ...")

                        print("Try Han's approach with EMU partition approach ...")
                        aff_han = han.sched_han(misc.average_case_convertor_taskset(tasksets_pure[s], avg_ratio), misc.average_case_convertor_typed(tasksets_typed[s], avg_ratio), tolerate_pa, tolerate_pb, 1)
                        if aff_han[0]:
                            affinities_han = defaultdict(int, sorted(aff_han[1].items(), key=lambda x: x[0]))
                            max_a_han, max_b_han = misc.find_max_ab_index(affinities_han)
                            unused_a_cores_han = processor_a - max_a_han - 1
                            if unused_a_cores_han > 0:
                                affinities_han = misc.adjust_unused_cores(affinities_han, unused_a_cores_han)
                            used_a_han = max_a_han + 1
                            used_b_han = max_b_han - processor_a + 1
                            print("Successfully pass the schedulability test, require core number: ", used_a_han,
                                  used_b_han)

                        print("Try improved federated scheduling approach ...")
                        aff_imp = imp_fed.improved_federated_p3(misc.average_case_convertor_taskset(tasksets_pure[s], avg_ratio), misc.average_case_convertor_typed(tasksets_typed[s], avg_ratio), tolerate_pa,
                                                                tolerate_pb, rho_imp_fed)
                        if aff_imp[0]:
                            affinities_imp = defaultdict(int, sorted(aff_imp[1].items(), key=lambda x: x[0]))
                            max_a_imp, max_b_imp = misc.find_max_ab_index(affinities_imp)
                            unused_a_cores_imp = processor_a - max_a_imp - 1
                            if unused_a_cores_imp > 0:
                                affinities_imp = misc.adjust_unused_cores(affinities_imp, unused_a_cores_imp)
                            used_a_imp = max_a_imp + 1
                            used_b_imp = max_b_imp - processor_a + 1
                            print("Successfully pass the schedulability test, require core number: ", used_a_imp,
                                  used_b_imp)

                        if aff_han[0] or aff_imp[0]:
                            affinity = []
                            if aff_han[0] and not aff_imp[0]:
                                affinity.append(affinities_han)
                                affinity.append([used_a_han, used_b_han])
                            elif not aff_han[0] and aff_imp[0]:
                                affinity.append(affinities_imp)
                                affinity.append([used_a_imp, used_b_imp])
                            else:
                                if used_a_han + used_b_han <= used_a_imp + used_b_imp:
                                    affinity.append(affinities_han)
                                    affinity.append([used_a_han, used_b_han])
                                else:
                                    affinity.append(affinities_imp)
                                    affinity.append([used_a_imp, used_b_imp])

                            aff_name = '../experiments/outputs/affinity_allocation/aff_tolerate_acet_' + str(
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
                            np.save(aff_name, np.array(affinity, dtype=object))
                            print("Optimized affinity information with acet and tolerate A and B processors has been saved.")

                        else:
                            print("It is not possible to generate feasible affinities by using current configurations.")
                            print("Try to generate a raw affinities by using type aware global schedule...")
                            affinity = []
                            affinity_raw = raw.gen_affinities_raw(tasksets[s], processor_a, processor_b)
                            affinity.append(affinity_raw)
                            affinity.append([processor_a, processor_b])
                            aff_name = '../experiments/outputs/affinity_allocation/aff_raw_wcet_' + str(
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
                            np.save(aff_name, np.array(affinity, dtype=object))
                            print("Raw affinity information with type aware global schedule has been saved.")

if __name__ == "__main__":
    main(sys.argv[1:])