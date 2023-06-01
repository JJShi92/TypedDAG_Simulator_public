# TypedDAG_Simulator_public
The simulator to schedule typed DAG tasks on heterogeneous multiprocessor platform.
<br />
## Before starting
The Dirichlet-Rescale (DRS) algorithm[^1] is applied for generate utilizations of task sets randomly
```
pip3 install drs
```
## Generators
Inside the genrators folder, we provide tools to 1) generate the configuration file; 2) generate pure DAG tasks; 3) generate the typed information, i.e., each node is assigned to one type of processors, and 4) generate the requested data address for each node.
#### DAG Task generator
The configuration file contains the following parameters:
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
# num_freq_data: number of the frequently requested data
# percent_freq: the percentage of requesting the frequently requested data
# allow_freq-0: fully randomly generate the requested data regardless of the frequently requested data
# allow_freq-1: control the percent of frequently requested data

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

After configure all of these aforementioned parameters in the file `configuration_generator.py`\\
Run `./generate_tasksets.sh` to generate all the needed information for task sets.

## Algorithms
Inside the `algorithms` folder, all these algorithms related files are included as follows:
- `affinity_improved_fed.py`: Two algorithms proposed in [^2]: a) the improved type-aware federated scheduling (Sec. 7), and b) the greedy type-aware federated scheduling algorithm (Sec. 5.2).
- `affinity_han.py`: The federated scheduling algorithms proposed by Han et al. in [^3], both `EMU` mode and `Greedy` mode are implemented.
- `affinity_raw.py`: The raw affinity generator by using type-aware global scheduling without any optimization and guarantee.
- `sched_sim.py`: The simulator to schedule operations of a given task set and given affinities.
- `misc.py`: Some helper functions that will be used globally.

## Experiments
In the `experiments` folder, we provide the tools to 
- `gen_affinity.py`: Generate the affinity of a given task set. 
- `gen_schedule.py`: Generate the schedule of a given task set with the generated optimizaed affinity.
- `draw_schedule.py`: The script to draw the Gantt chart of a generated schedule.
In the original mode, we only try Han's approach with EMU partition approach and our improved federated approach due to the better performance.
The Greedy partition approach for Han's approach and Greedy federated approach in [^2] are also supported but not applied in the `gen_affinity.py`.
<\br>
The final affinity will be selected according to the minial total required number of cores. 
If both of them are not feasible with the given WCET, we will try average case execution time (ACET) with the average ratio, minimal ratio and standard deviation.
And/or reset the upper bound of available number of processors of both types (to the tolerate bound) to check how many cores are needed with WCET and with ACET if WCET is still infeasible.
If non of aforementioned approaches is feasible, a raw affinity is assigned, i.e., type-aware global schedule, where all the cores of one type can be applied to execute any task if the corresponding node has the same typed assignment.

## Figures


## References
[^1]: https://pypi.org/project/drs/ 
[^2]: C. Lin, J. Shi, N. Ueter, M. Günzel, J. Reineke, and J. Chen. Type-aware federated scheduling for typed DAG tasks on heterogeneous multicore platforms. IEEE Trans. Computers, 72(5):1286–1300, 2023.
[^3]: M. Han, T. Zhang, Y. Lin, and Q. Deng. Federated scheduling for typed DAG tasks scheduling analysis on heterogeneous multi-cores. J. Syst. Archit., 112:101870, 2021.
