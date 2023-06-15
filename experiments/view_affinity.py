# The script to generate the schedule of given configurations

from __future__ import division
import numpy as np
import os
import math
import json
from collections import defaultdict
import sys
import getopt
sys.path.append('../../../')
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

    aff_file_name = 'aff_acet_2_0_[50, 100]_p16_8_q[0.2, 0.5]_u0.3_1_s0_6_5_m100_t1_1_0_d100_20_0.1_0'


    try:
        opts, args = getopt.getopt(argv, "hi:", ["conf_file_name", "aff_mod"])
    except getopt.GetoptError:
        print('view_affinity.py -i <the input affnity file name in npy format>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('view_affinity.py -i <the input affnity file name in npy format>')
            sys.exit()
        elif opt in ("-i", "--affname"):
            aff_file_name = str(arg)

        aff_file_name_np = aff_file_name + '.npy'
        aff_file_name_json = aff_file_name + '.json'

        if os.path.exists(aff_file_name_np):
            affinities, processors = np.load(aff_file_name_np, allow_pickle=True)

            total_info = []

            processors_info = {"Processor A": processors[0], "Processor B": processors[1]}

            total_info.append(processors_info)

            for i in range(len(affinities)):
                aff_info_temp = {"task id": i, "Porcessor A id": affinities[i][1], "Porcessor B id": affinities[i][2]}
                total_info.append(aff_info_temp)

            with open(aff_file_name_json, 'w') as json_file:
                json.dump(total_info, json_file, indent=4)

            print(total_info)
            print("The viewable file has been convert to " + str(aff_file_name_json))

        else:
            print("There is no feasible affinity file, please enter the correct aff file name with out .npy")

if __name__ == "__main__":
    main(sys.argv[1:])