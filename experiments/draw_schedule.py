from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def main(argv):
    # default json configuration file name
    conf_file_name = '../generators/configure.json'
    aff_mod = 0
    max_time = 10 ** 9
    set_id = 0
    time_bound = 200
    aff_mod = "raw"
    sched_file_name = None
    utilization = None
    try:
        opts, args = getopt.getopt(argv, "hi:s:m:t:f:u:", ["conf_file_name", "set_id", "aff_mod", "time_bound", "sched_file_name", "utilization"])
    except getopt.GetoptError:
        print('read_configuration.py -i <the JSON configuration file name> -s <set id> -m <aff mod> -t <time bound> -f <sched_file_name> -u <utilization>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('read_configuration.py -i <the JSON configuration file name> -s <set id> -m <aff mod> -t <time bound> -f <sched_file_name> -u <utilization>')
            sys.exit()
        elif opt in ("-i", "--conffname"):
            conf_file_name = str(arg)
        elif opt in ("-s", "--sid"):
            set_id = int(arg)
        elif opt in ("-m", "--mod"):
            aff_mod = str(arg)
        elif opt in ("-t", "--tbound"):
            time_bound = int(arg)
        elif opt in ("-f", "--sfname"):
            sched_file_name = str(arg)
        elif opt in ("-u", "--utilization"):
            utilization = int(arg)

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
    num_data_per_vertex = conf['num_data_per_vertex'][0]
    num_freq_data = conf['num_freq_data'][0]
    percent_freq = conf['percent_freq'][0]
    allow_freq = conf['allow_freq'][0]
    data_req_prob = conf['data_req_prob']

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

    if sched_file_name:
        if os.path.exists(sched_file_name):
            sched_results = np.load(sched_file_name, allow_pickle=True)
        else:
            print("The input schedule file name is incorrect, please check!")
            return
    else:
        print("Reading the schedule information ...")
        if utilization:
            utili = float(utilization / 100)
        else:
            ut = 0
            utili = float(util_all[ut] / 100)
        # original task set
        sched_wcet_name = '../experiments/outputs/schedule/sched_wcet_' + str(msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(set_id) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'

        if os.path.exists(sched_wcet_name):
            sched_results = np.load(sched_wcet_name, allow_pickle=True)
        else:
            sched_acet_name = '../experiments/outputs/schedule/sched_acet_' + str(msets) + '_' + str(
                ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                pc_prob) + '_u' + str(utili) + '_' + str(set_id) + '_s' + str(sparse) + '_' + str(
                int(math.log10(scale))) + '_' + str(
                preempt_times) + '_m' + str(
                main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                one_type_only) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'
            if os.path.exists(sched_acet_name):
                sched_results = np.load(sched_acet_name, allow_pickle=True)
            else:
                sched_tol_wcet_name = '../experiments/outputs/schedule/sched_tol_wcet_' + str(
                    msets) + '_' + str(
                    ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(processor_b) + '_q' + str(
                    pc_prob) + '_u' + str(utili) + '_' + str(set_id) + '_s' + str(sparse) + '_' + str(
                    int(math.log10(scale))) + '_' + str(
                    preempt_times) + '_m' + str(
                    main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                    one_type_only) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'
                if os.path.exists(sched_tol_wcet_name):
                    sched_results = np.load(sched_tol_wcet_name, allow_pickle=True)
                else:
                    sched_tol_acet_name = '../experiments/outputs/schedule/sched_tol_acet_' + str(
                        msets) + '_' + str(
                        ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                        processor_b) + '_q' + str(
                        pc_prob) + '_u' + str(utili) + '_' + str(set_id) + '_s' + str(sparse) + '_' + str(
                        int(math.log10(scale))) + '_' + str(
                        preempt_times) + '_m' + str(
                        main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                        one_type_only) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'
                    if os.path.exists(sched_tol_acet_name):
                        sched_results = np.load(sched_tol_acet_name, allow_pickle=True)
                    else:
                        sched_raw_name = '../experiments/outputs/schedule/sched_raw_' + str(
                            msets) + '_' + str(
                            ntasks) + '_' + str(num_nodes) + '_p' + str(processor_a) + '_' + str(
                            processor_b) + '_q' + str(
                            pc_prob) + '_u' + str(utili) + '_' + str(set_id) + '_s' + str(sparse) + '_' + str(
                            int(math.log10(scale))) + '_' + str(
                            preempt_times) + '_m' + str(
                            main_mem_time) + '_t' + str(skewness) + '_' + str(per_heavy) + '_' + str(
                            one_type_only) + '_d' + str(num_data_all) + '_' + str(num_data_per_vertex) + '_' + str(num_freq_data) + '_' + str(percent_freq) + '_' + str(data_req_prob) + '_' + str(allow_freq) + '.npy'
                        if os.path.exists(sched_raw_name):
                            sched_results = np.load(sched_raw_name, allow_pickle=True)
                        else:
                            print("The schedule does not exist, please check!")
                            return

    colors = [
        'b',  # Blue
        'g',  # Green
        'r',  # Red
        'c',  # Cyan
        'm',  # Magenta
        'y',  # Yellow
        'k',  # Black
        'darkblue',  # Dark Blue
        'darkgreen',  # Dark Green
        'darkred',  # Dark Red
        'darkcyan',  # Dark Cyan
        'darkmagenta',  # Dark Magenta
        'darkgoldenrod',  # Dark Yellow
        'darkkhaki',
        'darksalmon',
        'darkseagreen',
        'darkturquoise',
        'darkorchid',
        'darkviolet',
        'pink',
        'lightblue',  # Light Blue
        'lightgreen',  # Light Green
        'lightcoral',  # Light Red
        'lightcyan',  # Light Cyan
        'lightpink',  # Light Magenta
        'lightyellow'  # Light Yellow
        'lightgray',
        'coral',
        'olivedrab',
        'yellowgreen',
        'forestgreen',
        'limegreen',
        'mediumspringgreen',
        'lightseagreen',
        'darkslategrey',
        'powderblue',
        'dodgerblue',
        'slategray',
        'royalblue',
        'navy',
        'mediumslateblue',
        'rebeccapurple',
        'plum',
        'deeppink',
        'crimson'
    ]

    '''
    colors = [
    "blue", "green", "red", "cyan", "magenta", "yellow", "black", "white", "gray", "darkblue",
    "darkgreen", "darkred", "darkcyan", "darkmagenta", "darkyellow", "darkgray", "lightblue",
    "lightgreen", "lightred", "lightcyan", "lightmagenta", "lightyellow", "lightgray", "navy",
    "forestgreen", "maroon", "teal", "purple", "olive", "silver", "skyblue", "limegreen", "tomato",
    "aqua", "orchid", "gold", "dimgray", "royalblue", "mediumseagreen", "indianred", "cadetblue",
    "mediumorchid", "khaki", "slategray", "deepskyblue", "mediumaquamarine", "rosybrown",
    "lightsteelblue", "palegreen", "darkorange"
    ]
    '''

    fig, ax = plt.subplots()

    legend_patches = []  # List to store legend patches
    plotted_tasks = set()  # Set to track the tasks that have been plotted

    durations = 0

    # Plot the bars for each task on the processors
    for i, processor in enumerate(sched_results[0]):
        for j, task_info in enumerate(processor.schedule_record):
            task_id, vertex_id, deadline, rest_execution, start_time, finish_time = task_info
            duration = finish_time - start_time
            durations += duration

            start_time = start_time/scale
            finish_time = finish_time/scale
            duration = duration/scale

            durations += duration

            # Plot the task bar
            ax.barh(i, duration, left=start_time, height=0.5, color=colors[task_id % len(colors)])

            # Store the legend patch for each unique task
            if task_id not in plotted_tasks:
                legend_patch = mpatches.Patch(color=colors[task_id % len(colors)], label=f"Task {task_id}")
                legend_patches.append(legend_patch)
                plotted_tasks.add(task_id)


            if finish_time > time_bound:
                break

    print("Total_utilization: ", durations/12000000000)

    legend_patches = sorted(legend_patches, key=lambda patch: int(patch.get_label().split()[1]))

    # Set labels and formatting
    ax.set_yticks(np.arange(len(sched_results[0])))
    ax.set_yticklabels([f"Processor {i + 1}" for i in range(len(sched_results[0]))])
    ax.set_xlabel("Time")
    ax.set_ylabel("Processor")
    ax.set_title("Gantt Chart - Task Schedule")

    plt.grid(axis='x')  # Add grid lines on the x-axis

    # Add legend
    plt.legend(handles=legend_patches, loc='upper right')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

