# TypedDAG_Simulator_public
The simulator to schedule typed DAG tasks on heterogeneous multiprocessor platform.
<br />
## Before starting
The Dirichlet-Rescale (DRS) algorithm[^1] is applied to generate utilizations of task sets randomly
```
pip3 install drs
```
## Generators
Inside the genrators folder, we provide tools to 1) generate the configuration file; 2) generate pure DAG tasks; 3) generate the typed information, i.e., each node is assigned to one type of processors, and 4) generate the requested data address for each node.
#### DAG Task generator
The configuration file contains the following parameters:
| Parameter        | Description                                                                                                  |
|------------------|--------------------------------------------------------------------------------------------------------------|
| msets            | Number of sets                                                                                               |
| ntasks           | Number of tasks for each set. If `ntasks < 1`, the number of tasks is generated randomly due to the sparse parameter |
| spars-0          | Range: `[0.5 * max(aprocessor, bprocessor), 2 * max(aprocessor, bprocessor)]`                                |
| spars-1          | Range: `[(aprocessor + aprocessor), 2 * (aprocessor + aprocessor)]`                                          |
| spars-2          | Range: `[0.25 * (aprocessor + aprocessor), (aprocessor + aprocessor)]`                                      |
| aprocessor       | Number of processor A                                                                                        |
| bprocessor       | Number of processor B                                                                                        |
| pc_prob          | Lower and upper bounds of the probability of two vertices having an edge. The real probability is in the range `[pc_prob_l, pc_prob_h]` |
| utilization      | Total utilization for a set of tasks                                                                          |
| scale            | Scale to keep all the parameters as integers                                                                  |
| skewness         | Controls the skewness of the skewed tasks                                                                     |
| per_heavy        | Percentage of heavy^a or heavy^b tasks (e.g., 0%, 25%, 50%, 75%, and 100%)                                   |
| one_type_only    | Whether to allow a task to require only one type of processor: 0 (not allowed) or 1 (allowed)                |
| num_data_all     | Number of all available data                                                                                  |
| num_freq_data    | Number of frequently requested data                                                                           |
| percent_freq     | Percentage of requesting the frequently requested data                                                       |
| allow_freq-0     | Generate requested data randomly regardless of the frequently requested data                                 |
| allow_freq-1     | Control the percentage of frequently requested data                                                          |
| main_mem_size    | Size of the main memory. Assume a very large number can store all the requested data                          |
| main_mem_time    | Time for data access from main memory                                                                         |
| fast_mem_size    | Size of the fast memory                                                                                        |
| fast_mem_time    | Time for data access from fast memory                                                                          |
| l1_cache_size    | Size of the L1 cache                                                                                          |
| l1_cache_time    | Time for data access from L1 cache                                                                             |
| try_avg_case     | Whether to try average case execution time (acet) when the WCET cannot pass the schedulability test          |
| avg_ratio        | Ratio of acet/wcet                                                                                            |
| std_dev          | Standard deviation for generating real case execution time when simulating the schedule                      |
| tolerate_pa      | Upper bound of the tolerable number of type A processors when the current number is not enough               |
| tolerate_pb      | Upper bound of the tolerable number of type B processors when the current number is not enough               |
| rho_greedy       | Setting of rho for the greedy federated schedule algorithm. `0 < rho <= 0.5`                                  |
| rho_imp_fed      | Setting of rho for the improved federated schedule algorithm. `rho = 1/7.25`                                 |


After configure all of these aforementioned parameters in the file `configuration_generator.py`
<br />
Run `./generate_tasksets.sh` to generate all the needed information for task sets.
- `/experiments/inputs/tasks_pure` stores the pure task sets.
- `/experiments/inputs/tasks_typed` stores the typed allocation for each vertex from all tasks.
- `/experiments/inputs/tasks_data_request` stores the requested data addresses of each vertex from all tasks

## Algorithms
Inside the `algorithms` folder, all these algorithms related files are included as follows:
- `affinity_improved_fed.py`: Two algorithms proposed in [^2]: a) the improved type-aware federated scheduling (Sec. 7), and b) the greedy type-aware federated scheduling algorithm (Sec. 5.2).
- `affinity_han.py`: The federated scheduling algorithms proposed by Han et al. in [^3], both `EMU` mode and `Greedy` mode are implemented.
- `affinity_raw.py`: The raw affinity generator by using type-aware global scheduling without any optimization and guarantee.
- `sched_sim.py`: The simulator to schedule operations of a given task set and given affinities.
- `misc.py`: Some helper functions that will be used globally.

## Experiments
In the `experiments` folder, we provide the tools to 
- `gen_affinity.py`: Generate the affinity of a given task set in `/experiments/outputs/affinity_allocation/`. 
- `gen_schedule.py`: Generate the schedule of a given task set with the generated optimizaed affinity in `/experiments/outputs/schedule`.
- `draw_schedule.py`: The script to draw the Gantt chart of a generated schedule.

### Please Note: 
Once new task sets are generated with the same configuration file, the affinity_allocation information from the previous task sets (with the same configurations) has to be cleaned.

In the original mode, we only try Han's approach with EMU partition approach and our improved federated approach due to the better performance.
The Greedy partition approach for Han's approach and Greedy federated approach in [^2] are also supported but not applied in the `gen_affinity.py`.

The final affinity will be selected according to the minial total required number of cores. 
If both of them are not feasible with the given WCET, we will try:
- Average case execution time (ACET) with the average ratio, minimal ratio and standard deviation.
- And/or reset the upper bound of available number of processors of both types (to the tolerate bound) to check how many cores are needed with WCET and with ACET if WCET is still infeasible.
- If non of aforementioned approaches is feasible, a raw affinity is assigned, i.e., type-aware global schedule, where all the cores of one type can be applied to execute any task if the corresponding node has the same typed assignment.

## Figures


## References
[^1]: https://pypi.org/project/drs/ 
[^2]: C. Lin, J. Shi, N. Ueter, M. Günzel, J. Reineke, and J. Chen. [Type-aware federated scheduling for typed DAG tasks on heterogeneous multicore platforms.](https://ieeexplore.ieee.org/document/9869701){target="_blank"} IEEE Trans. Computers, 72(5):1286-1300, 2023.
[^3]: M. Han, T. Zhang, Y. Lin, and Q. Deng. Federated scheduling for typed DAG tasks scheduling analysis on heterogeneous multi-cores. J. Syst. Archit., 112:101870, 2021.
