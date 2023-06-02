# The improvde federated scheduling algorithm for DAG tasks on heterogeneous multi-core systems
import math
import copy
import time
from collections import deque
from collections import defaultdict
import itertools
import sys
sys.path.append('../')
from algorithms import affinity_han as han
from algorithms import misc


# suspension time processor a
# task_org = [task, type_info]
def suspension_a(task_org, processor_b):
    task = copy.deepcopy(task_org)
    s_a = task[1].cpB + ((task[1].utilizationB * task[0].period - task[1].cpB) / processor_b)
    return s_a


# suspension time processor b
# task_org = [task, type_info]
def suspension_b(task_org, processor_a):
    task = copy.deepcopy(task_org)
    s_b = task[1].cpA + ((task[1].utilizationA * task[0].period - task[1].cpA) / processor_a)
    return s_b


# the sum of ceiling for task with higher priorities: heavy_a
# task_hp = [[task, typed, R_i]...]
def sum_hp_a(time_t, tasks_hp):
    sum = 0
    for i in range(len(tasks_hp)):
        sum = sum + math.ceil((((time_t + tasks_hp[i][1]) / tasks_hp[i][0][0].period) - tasks_hp[i][0][1].utilizationB)) * tasks_hp[i][0][1].utilizationB * tasks_hp[i][0][0].period
    return sum


# the sum of ceiling for task with higher priorities: heavy_b
# task_hp = [[task, typed, R_i]...]
def sum_hp_b(time_t, tasks_hp):
    sum = 0
    for i in range(len(tasks_hp)):
        sum = sum + math.ceil((((time_t + tasks_hp[i][1]) / tasks_hp[i][0][0].period) - tasks_hp[i][0][1].utilizationA)) * tasks_hp[i][0][1].utilizationA * tasks_hp[i][0][0].period
    return sum


# schedulability heavy a on 1 B core
# task_hp = [[task, typed, R_i]...]
# task_new = [task, typed]
def sched_heavy_a(task_new, tasks_hp, processor_a):
    if processor_a <= 0:
        return False
    constant_cs = task_new[1].utilizationB * task_new[0].period + suspension_b(task_new, processor_a)
    # the first task on processor A
    if len(tasks_hp) == 0:
        return constant_cs

    time_t = 1
    response_time = 0
    start_time = time.time()
    #while time_t <= task_new[0].period and time.time() - start_time < 1200:
    while time_t <= task_new[0].period:
        response_time = constant_cs + sum_hp_a(time_t, tasks_hp)

        if response_time <= time_t:
            return response_time
        else:
            time_t = response_time
    return False


# schedulability heavy b on 1 A core
# task_hp = [[task, typed, R_i]...]
# task_new = [task, typed]
def sched_heavy_b(task_new, tasks_hp, processor_b):
    if processor_b <= 0:
        return False
    constant_cs = task_new[1].utilizationA * task_new[0].period + suspension_a(task_new[0], processor_b)
    # the first task on processor A
    if len(tasks_hp) == 0:
        return constant_cs

    time_t = 1
    response_time = 0
    start_time = time.time()
    # while time_t <= task_new[0].period and time.time() - start_time < 1200:
    while time_t <= task_new[0].period:
        response_time = constant_cs + sum_hp_b(time_t, tasks_hp)

        if response_time <= time_t:
            return response_time
        else:
            time_t = response_time
    return False


# the processors A/B for heavy_ab task
# heavy_ab = [task, typed]
def processors_heavy_ab(heavy_ab):
    p_ab = []
    p_a = math.ceil((heavy_ab[1].utilizationA * heavy_ab[0].period - heavy_ab[1].cpA) / (heavy_ab[0].period / 2 - heavy_ab[1].cpA))
    p_ab.append(p_a)
    p_b = math.ceil((heavy_ab[1].utilizationB * heavy_ab[0].period - heavy_ab[1].cpB) / (heavy_ab[0].period / 2 - heavy_ab[1].cpB))
    p_ab.append(p_b)
    return p_ab


# the processor A for heavy_a task
# heavy_a = [heavy_a_task, typed]
def processors_heavy_a(heavy_a):
    p_a = math.ceil((heavy_a[1].utilizationA * heavy_a[0].period - heavy_a[1].cpA) / (heavy_a[0].period / 3 - heavy_a[1].cpA))
    return p_a


# the processor B for heavy_b task
# heavy_b = [heavy_b_task, typed]
def processors_heavy_b(heavy_b):
    p_b = math.ceil((heavy_b[1].utilizationB * heavy_b[0].period - heavy_b[1].cpB) / (heavy_b[0].period / 3 - heavy_b[1].cpB))
    return p_b


# schedule heavy_a on B core along with other tasks
def sched_share_heavy_a(heavy_a, partitioned_b_org, processor_a):
    partitioned_b = copy.deepcopy(partitioned_b_org)
    for i in range(len(partitioned_b)):
        response = sched_heavy_a(heavy_a, partitioned_b[i], processor_a)
        if response:
            new_partition = []
            new_partition.append(heavy_a)
            new_partition.append(response)
            partitioned_b[i].append(new_partition)
            return True, partitioned_b, i

    return False, partitioned_b, -1


# schedule heavy_b on A core along with other tasks
def sched_share_heavy_b(heavy_b, partitioned_a_org, processor_b):
    partitioned_a = copy.deepcopy(partitioned_a_org)
    for i in range(len(partitioned_a)):
        response = sched_heavy_b(heavy_b, partitioned_a[i], processor_b)
        if response:
            new_partition = []
            new_partition.append(heavy_b)
            new_partition.append(response)
            partitioned_a[i].append(new_partition)
            return True, partitioned_a, i

    return False, partitioned_a, -1


# schedule a light task on one a core and one b core
def sched_light_fix(light_task, task_hp_a, task_hp_b):
    wcet_new = sum(light_task[0].weights)
    # if both a and b processor is empty
    if len(task_hp_a) == 0 and len(task_hp_b) == 0:
        return wcet_new

    time_t = 1
    response_time = 0
    start_time = time.time()
    # while time_t <= light_task.period and time.time() - start_time < 1200:
    # it requires both A and B core
    if light_task[1].utilizationA > 0 and light_task[1].utilizationB > 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_a(time_t, task_hp_b) + sum_hp_b(time_t, task_hp_a)

            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    # it only requires A core
    elif light_task[1].utilizationA > 0 and light_task[1].utilizationB == 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_b(time_t, task_hp_a)
            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    # it only requires B core
    elif light_task[1].utilizationA == 0 and light_task[1].utilizationB > 0:
        while time_t <= light_task[0].period:
            response_time = wcet_new + sum_hp_a(time_t, task_hp_b)

            if response_time <= time_t:
                return response_time
            else:
                time_t = response_time

    return False


# schedule light task on A/B core along with other tasks
# light_ab = [task, type_info]
def sched_share_light(light_ab, partitioned_ab_org):
    partitioned_ab = copy.deepcopy(partitioned_ab_org)

    for i in range(len(partitioned_ab[0])):
        for j in range(len(partitioned_ab[1])):
            response = sched_light_fix(light_ab, partitioned_ab[0][i], partitioned_ab[1][j])
            if response:
                new_partition_a = []
                new_partition_a.append(light_ab)
                new_partition_a.append(response)
                new_partition_b = []
                new_partition_b.append(light_ab)
                new_partition_b.append(response)

                if light_ab[1].utilizationA > 0 and light_ab[1].utilizationB > 0:
                    partitioned_ab[0][i].append(new_partition_a)
                    partitioned_ab[1][j].append(new_partition_b)
                    return True, partitioned_ab, [i, j]
                elif light_ab[1].utilizationA > 0 and light_ab[1].utilizationB == 0:
                    partitioned_ab[0][i].append(new_partition_a)
                    return True, partitioned_ab, [i, -1]
                elif light_ab[1].utilizationA == 0 and light_ab[1].utilizationB > 0:
                    partitioned_ab[1][j].append(new_partition_b)
                    return True, partitioned_ab, [-1, i]

    return False, partitioned_ab, [-1, -1]


# calculate the number of processor A if only processor A is required
# task_org = [task, type_info]
def only_processor_a(task_org, available_a):
    task = copy.deepcopy(task_org)
    current_available_a = copy.deepcopy(available_a)
    for i in range(1, current_available_a+1):
        if han.hgsh(task[0], task[1], [i, 0]) <= task[0].period:
            return int(i)
    return 0


# calculate the number of processor B if only processor B is required
# task_org = [task, type_info]
def only_processor_b(task_org, available_b):
    task = copy.deepcopy(task_org)
    current_available_b = copy.deepcopy(available_b)
    for i in range(1, current_available_b+1):
        if han.hgsh(task[0], task[1], [0, i]) <= task[0].period:
            return int(i)
    return 0


# Greedy federated partitioned algorithm
def greedy_federated_p(task_set, typed_org, rho, processor_a, processor_b):
    tasks = copy.deepcopy(task_set)
    type_info = copy.deepcopy(typed_org)
    tasks_share = deque()
    available_a = copy.deepcopy(processor_a)
    available_b = copy.deepcopy(processor_b)

    # Record the current index for both processor A and B under allocation
    # current_index = [current_a_index, current_b_index]
    current_index = []
    current_a_index = 0
    current_b_index = copy.deepcopy(processor_a)
    current_index.append(current_a_index)
    current_index.append(current_b_index)

    # Affinities for tasks
    affinities = defaultdict(list)

    # used A/B processors
    used_a = 0
    used_b = 0

    # clarify tasks into four groups
    for i in range(len(tasks)):

        # check if the task only require one type of processor
        # only require processor A:
        if task_set[i][1].utilizationB == 0 and task_set[i][1].utilizationA > 1:
            if available_a > 0:
                used_a = only_processor_a([tasks[i], type_info[i]], available_a)
            else:
                return False, affinities, [-1, 0]
            if 0 < used_a <= available_a:
                affinities[i], current_index = misc.assign_affinity_task(current_index, [used_a, 0])
                available_a = available_a - used_a
            else:
                return False, affinities, [-1, 0]

        # only require processor B:
        if task_set[i][1].utilizationA == 0 and task_set[i][1].utilizationB > 1:
            if available_b > 0:
                used_b = only_processor_b([tasks[i], type_info[i]], available_b)
            else:
                return False, affinities, [0, -1]
            if 0< used_b <= available_b:
                affinities[i], current_index = misc.assign_affinity_task(current_index, [0, used_b])
                available_b = available_b - used_b
            else:
                return False, affinities, [0, -1]

        if type_info[i].utilizationA > rho:
            if type_info[i].utilizationB > rho:
                # heavy_ab
                used_ab = processors_heavy_ab([tasks[i], type_info[i]])
                if used_ab[0] < 0 or used_ab[1] < 0:
                    return False, affinities, used_ab
                used_a += used_ab[0]
                used_b += used_ab[1]
                affinities[i], current_index = misc.assign_affinity_task(current_index, used_ab)
            else:
                # heavy_a
                new_used_a = processors_heavy_a([tasks[i], type_info[i]])
                if new_used_a < 0:
                    return False, affinities, [new_used_a, 0]
                else:
                    affinities[i], current_index = misc.assign_affinity_task(current_index, [new_used_a, 0])
                    used_a += new_used_a
                tsk_temp = []
                tsk_temp.append([tasks[i], type_info[i]])
                tsk_temp.append(1)
                tsk_temp.append(new_used_a)
                tasks_share.append(tsk_temp)
        else:
            if type_info[i].utilizationB > rho:
                # heavy_b
                new_used_b = processors_heavy_b([tasks[i], type_info[i]])
                if new_used_b < 0:
                    return False, affinities, [0, new_used_b]
                else:
                    affinities[i], current_index = misc.assign_affinity_task(current_index, [0, new_used_b])
                    used_b += new_used_b
                tsk_temp = []
                tsk_temp.append([tasks[i], type_info[i]])
                tsk_temp.append(2)
                tsk_temp.append(new_used_b)
                tasks_share.append(tsk_temp)
            else:
                tsk_temp = []
                tsk_temp.append([tasks[i], type_info[i]])
                tsk_temp.append(3)
                tasks_share.append(tsk_temp)

    # judge if the used ab is larger than available ab processors
    if used_a > available_a or used_b > available_b:
        return False, affinities, [available_a - used_a, available_b - used_b]

    available_a = available_a - used_a
    available_b = available_b - used_b

    if len(tasks_share) == 0:
        return True, affinities, current_index

    if len(tasks_share) == 1:
        if tasks_share[0][0][1].utilizationA > 0 and tasks_share[0][0][1].utilizationB == 0 and available_a > 0:
            # schedulable by default
            affinities[tasks_share[0][0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [1, 0])
            return True, affinities, current_index
        elif tasks_share[0][0][1].utilizationB > 0 and tasks_share[0][0][1].utilizationA == 0 and available_b > 0:
            # schedulable by default
            affinities[tasks_share[0][0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [0, 1])
            return True, affinities, current_index
        elif tasks_share[0][0][1].utilizationB > 0 and tasks_share[0][0][1].utilizationA > 0 and available_a > 0 and available_b > 0:
            # schedulable by default
            affinities[tasks_share[0][0][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [1, 1])
            return True, affinities, current_index
        else:
            return False, affinities, [available_a - 1, available_b - 1]

    partition_ab = []
    partition_a = []
    partition_b = []
    for i in range(available_a):
        partition_a.append(deque())
    for i in range(available_b):
        partition_b.append(deque())

    partition_ab.append(partition_a)
    partition_ab.append(partition_b)

    shared_start_index = copy.deepcopy(current_index)

    # handle tasks_share
    # RM sort at first
    tasks_share = deque(sorted(tasks_share, key=lambda x: x[0][0].priority))
    for i in range(len(tasks_share)):
        # heavy_a
        if tasks_share[i][1] == 1:
            new_partitioned_b = sched_share_heavy_a(tasks_share[i][0], partition_ab[1], tasks_share[i][2])
            if new_partitioned_b[0]:
                affinities[tasks_share[i][0][0].tsk_id], current_index = misc.assign_affinity_single_heavy_task(current_index, [-1, new_partitioned_b[2] + shared_start_index[1]])
                partition_ab[1] = new_partitioned_b[1]
            else:
                return False, affinities, [0, -1]
        # heavy_b
        elif tasks_share[i][1] == 2:
            new_partitioned_a = sched_share_heavy_b(tasks_share[i][0], partition_ab[0], tasks_share[i][2])
            if new_partitioned_a[0]:
                affinities[tasks_share[i][0][0].tsk_id], current_index = misc.assign_affinity_single_heavy_task(current_index, [new_partitioned_a[2] + shared_start_index[0], -1])
                partition_ab[0] = new_partitioned_a[1]
            else:
                return False, affinities, [-1, 0]

        # light
        else:
            new_partitioned_ab = sched_share_light(tasks_share[i][0], partition_ab)

            if new_partitioned_ab[0]:
                partition_ab = new_partitioned_ab[1]
                used_shared_index = [x + y for x, y in zip(shared_start_index, new_partitioned_ab[2])]
                affinities[tasks_share[i][0][0].tsk_id], current_index = misc.assign_affinity_light_task(current_index, used_shared_index)
            else:
                return False, affinities, new_partitioned_ab[2]

    return True, affinities, current_index


# calculate the a cores according to the b suspension time
def heavy_a_cores(heavy_a, suspension_b):
    m_a = math.ceil((heavy_a[1].utilizationA * heavy_a[0].period - heavy_a[1].cpA) / (suspension_b - heavy_a[1].cpA))
    return m_a


# calculate the b cores according to the a suspension time
def heavy_b_cores(heavy_b, suspension_a):
    m_b = math.ceil((heavy_b[1].utilizationB * heavy_b[0].period - heavy_b[1].cpB) / (suspension_a - heavy_b[1].cpB))
    return m_b


def exclusive_heavy_a(heavy_a):
    upper_bound_a = processors_heavy_a(heavy_a)
    suspension_b = heavy_a[0].period * (1 - heavy_a[1].utilizationB)
    cores_needed = heavy_a_cores(heavy_a, suspension_b)
    if cores_needed <= upper_bound_a:
        return cores_needed
    else:
        return False


def exclusive_heavy_b(heavy_b):
    upper_bound_b = processors_heavy_b(heavy_b)
    suspension_a = heavy_b[0].period * (1 - heavy_b[1].utilizationA)
    cores_needed = heavy_b_cores(heavy_b, suspension_a)
    if cores_needed <= upper_bound_b:
        return cores_needed
    else:
        return False


# try to add new cores to schedule a new light task
def shared_light_newcore(light_task, partitioned_ab_org, available_cores):
    # try to add one new core(s)
    partitioned_a = copy.deepcopy(partitioned_ab_org[0])
    partitioned_b = copy.deepcopy(partitioned_ab_org[1])
    available_a = copy.deepcopy(available_cores[0])
    available_b = copy.deepcopy(available_cores[1])

    if available_a == 0 and available_b == 0:
        return False, [partitioned_a, partitioned_b], [-1, -1]

    # no task in the shared A core
    # only add a B core is sufficient
    # currently only the task in the A core
    if light_task[1].utilizationA > 0 and light_task[1].utilizationB > 0 and len(partitioned_a[0]) == 0:
        new_processor = []
        new_task = []
        new_task.append(light_task)
        new_task.append(light_task[0].utilization * light_task[0].period)
        new_processor.append(new_task)

        partitioned_a[0].append(new_task)

        partitioned_b.append(new_processor)

        # only use one new b core
        # The last [-1, -1] means the task does not share with other tasks
        # It does not use the existed cores
        return True, [partitioned_a, partitioned_b], [0, 1], [-1, -1]


    # no task in the shared B core
    # only add a A core is sufficient
    # currently only the task in the B core
    if light_task[1].utilizationB > 0 and light_task[1].utilizationA > 0 and len(partitioned_b[0]) == 0:
        new_processor = []
        new_task = []
        new_task.append(light_task)
        new_task.append(light_task[0].utilization * light_task[0].period)
        new_processor.append(new_task)

        partitioned_b[0].append(new_task)
        partitioned_a.append(new_processor)

        # only use one new a core
        # The last [-1, -1] means the task does not share with other tasks
        # It does not use the existed cores
        return True, [partitioned_a, partitioned_b], [1, 0], [-1, -1]

    # it is impossible that both a/b core are empty

    # only allocate one moe a core
    if available_a > 0 and available_a >= available_b:
        temp_partition_a = copy.deepcopy(partitioned_a)
        temp_partition_a.append([])

        temp_partition_ab = sched_share_light(light_task, [temp_partition_a, partitioned_b])
        if temp_partition_ab[0]:
            partitioned_a = copy.deepcopy(temp_partition_ab[1][0])
            partitioned_b = copy.deepcopy(temp_partition_ab[1][1])
            return True, [partitioned_a, partitioned_b], [1, 0], temp_partition_ab[2]

    # only allocate one more b core
    if available_b > 0:
        temp_partition_b = copy.deepcopy(partitioned_b)
        temp_partition_b.append([])

        temp_partition_ab = sched_share_light(light_task, [partitioned_a, temp_partition_b])
        if temp_partition_ab[0]:
            partitioned_a = copy.deepcopy(temp_partition_ab[1][0])
            partitioned_b = copy.deepcopy(temp_partition_ab[1][1])
            return True, [partitioned_a, partitioned_b], [0, 1], temp_partition_ab[2]

    # allocate two new a/b cores
    if available_a > 0 and available_b > 0:
        new_processor = []
        new_task = []
        new_task.append(light_task)
        new_task.append(light_task[0].utilization * light_task[0].period)
        new_processor.append(new_task)

        partitioned_a.append(new_processor)
        partitioned_b.append(new_processor)

        return True, [partitioned_a, partitioned_b], [1, 1], [-1, -1]

    # no sufficient cores are available
    else:
        return False, [partitioned_a, partitioned_b], [available_a - 1, available_b - 1]


# check the combination of a/b processors
# if the one of the combination is feasible
def validate_heavy_ab_raw(processor_ab, available_a, available_b, start_time):
    start_time_fix = copy.deepcopy(start_time)
    candidates = copy.deepcopy(processor_ab)
    a = copy.deepcopy(available_a)
    b = copy.deepcopy(available_b)

    # FIXME
    if time.time()-start_time > 1800000:
        return False

    if a < 0 or b < 0:
        return False

    if not candidates:
        return 1

    for selection in candidates[-1]:
        if validate_heavy_ab(candidates[:-1], a - selection[0], b - selection[1], start_time_fix):
            return 1
    return False


# check the combination of a/b processors
# using dynamic programming
# complexity O(m^n) (m is the maximum number of sublists in a row, and n is the number of rows in the big list)
def validate_heavy_ab_iter(processor_ab_org, available_a, available_b):
    processor_ab = copy.deepcopy(processor_ab_org)
    min_sum = 10000
    min_combination = None

    for combination in itertools.product(*processor_ab):
        first_sum = sum(item[0] for item in combination)
        second_sum = sum(item[1] for item in combination)

        if first_sum <= available_a and second_sum <= available_b:
            total_sum = first_sum + second_sum
            if total_sum < min_sum:
                min_sum = total_sum
                min_combination = combination

    if min_combination:
        return min_combination
    else:
        return False


# check the combination of a/b processors
# using dynamic programming
# complexity O(n * b1 * b2 * m) (n is the number of rows in the big list, b1 and b2 are the given bounds, and m is the average number of sublists in each row)
def validate_heavy_ab_dp(processor_ab_org, available_a, available_b):
    processor_ab = copy.deepcopy(processor_ab_org)

    n = len(processor_ab)  # Number of rows in the big list

    # Initialize the dynamic programming table
    dp = [[[float('inf')] * (available_b + 1) for _ in range(available_a + 1)] for _ in range(n + 1)]
    dp[0][0][0] = 0

    # Populate the dynamic programming table
    for i in range(1, n + 1):
        for j in range(available_a + 1):
            for k in range(available_b + 1):
                for sublist in processor_ab[i - 1]:
                    if sublist[0] <= j and sublist[1] <= k:
                        dp[i][j][k] = min(dp[i][j][k], dp[i - 1][j - sublist[0]][k - sublist[1]] + sublist[0] + sublist[1])

    # Find the minimum sum and the corresponding combination
    min_sum = float('inf')
    min_combination = None

    for j in range(available_a, -1, -1):
        for k in range(available_b, -1, -1):
            if dp[n][j][k] < min_sum:
                min_sum = dp[n][j][k]
                min_combination = (j, k)

    # If no valid combination found, return False
    # if min_combination == (available_a, available_b) or min_sum == float('inf'):
    if min_sum == float('inf'):
        return False

    # Reconstruct the combination from the dynamic programming table
    combination = []
    for i in range(n, 0, -1):
        sublist = None
        for sublist in processor_ab[i - 1]:
            if sublist[0] <= min_combination[0] and sublist[1] <= min_combination[1] and dp[i][min_combination[0]][min_combination[1]] == dp[i - 1][min_combination[0] - sublist[0]][min_combination[1] - sublist[1]] + sublist[0] + sublist[1]:
                combination.append(sublist)
                min_combination = (min_combination[0] - sublist[0], min_combination[1] - sublist[1])
                break

    # Sort the combination according to the original list order
    sorted_combination = []
    for sublist in processor_ab:
        for item in sublist:
            if item in combination:
                sorted_combination.append(item)

    return sorted_combination if sorted_combination else False


# find if the heavy^a task can share a b core
def find_share_b(heavy_a, partition_b_org, exclusive_a_cores):
    partition_b = copy.deepcopy(partition_b_org)

    new_partition = sched_share_heavy_a(heavy_a, partition_b, exclusive_a_cores)

    if new_partition:
        return new_partition
    else:
        return False


# find if the heavy^b task can share a b core
def find_share_a(heavy_b, partition_a_org, exclusive_b_cores):
    partition_a = copy.deepcopy(partition_a_org)

    new_partition = sched_share_heavy_b(heavy_b, partition_a, exclusive_b_cores)

    if new_partition:
        return new_partition
    else:
        return False


def sum_utilization_one_a(paration_a):
    tasks = copy.deepcopy(paration_a)
    sum_u = 0
    for i in range(len(tasks)):
        sum_u = sum_u + tasks[i][0][1].cpA

    return sum_u


def sum_utilization_one_b(paration_b):
    tasks = copy.deepcopy(paration_b)
    sum_u = 0
    for i in range(len(tasks)):
        sum_u = sum_u + tasks[i][0][1].cpB

    return sum_u


# find the minimal a cores for heavy_a
def minimal_heavy_a(heavy_a, partition_b_org, rho):
    partition_b = copy.deepcopy(partition_b_org)
    upper_bound_a = processors_heavy_a(heavy_a)

    temp_info = []

    for i in range(len(partition_b)):
        if sum_utilization_one_b(partition_b[i]) < rho:
            suspension_b = heavy_a[0].period * (1 - heavy_a[1].utilizationB) - sum_hp_a(heavy_a[0].period, partition_b[i])
            if suspension_b > 0:
                cores_needed = heavy_a_cores(heavy_a, suspension_b)
                if 0 < cores_needed <= upper_bound_a:
                    temp_info.append([i, cores_needed])

    # select the partition with minimal a core
    if len(temp_info) > 0:
        temp_info.sort(key=lambda x: x[1])
        temp_partition = []
        temp_partition.append(heavy_a)
        response = sched_heavy_a(heavy_a, partition_b[temp_info[0][0]], temp_info[0][1])
        temp_partition.append(response)
        partition_b[temp_info[0][0]].append(temp_partition)
        return [partition_b, temp_info[0]]
    else:
        return False


# find the minimal b cores for heavy_b
def minimal_heavy_b(heavy_b, partition_a_org, rho):
    partition_a = copy.deepcopy(partition_a_org)
    upper_bound_b = processors_heavy_b(heavy_b)
    temp_info = []

    for i in range(len(partition_a)):
        if sum_utilization_one_a(partition_a[i]) < rho:
            suspension_a = heavy_b[0].period * (1 - heavy_b[1].utilizationA) - sum_hp_b(heavy_b[0].period, partition_a[i])
            if suspension_a > 0:
                cores_needed = heavy_b_cores(heavy_b, suspension_a)
                if 0 < cores_needed <= upper_bound_b:
                    temp_info.append([i, cores_needed])

    # select the partition with minimal a core
    if len(temp_info) > 0:
        temp_info.sort(key=lambda x: x[1])
        temp_partition = []
        temp_partition.append(heavy_b)
        response = sched_heavy_a(heavy_b, partition_a[temp_info[0][0]], temp_info[0][1])
        temp_partition.append(response)
        partition_a[temp_info[0][0]].append(temp_partition)
        return [partition_a, temp_info[0]]
    else:
        return False


# improved federated partitioned scheduling 3
def improved_federated_p3(task_set_org, typed_org, processor_a, processor_b, rho):
    task_set_pure = copy.deepcopy(task_set_org)
    type_info = copy.deepcopy(typed_org)
    available_a = int(copy.deepcopy(processor_a))
    available_b = int(copy.deepcopy(processor_b))

    # Record the current index for both processor A and B under allocation
    # current_index = [current_a_index, current_b_index]
    current_index = []
    current_a_index = 0
    current_b_index = copy.deepcopy(processor_a)
    current_index.append(current_a_index)
    current_index.append(current_b_index)

    # Affinities for tasks
    affinities = defaultdict(list)

    # Combine the task set with the typed info
    task_set = []
    for i in range(len(task_set_pure)):
        task_set.append([task_set_pure[i], type_info[i]])

    # rm sort the task
    task_set.sort(key=lambda x: x[0].priority)

    heavy_ab = deque()
    partition_ab = []
    partitioned_a = deque()
    partitioned_b = deque()

    partition_ab.append(partitioned_a)
    partition_ab.append(partitioned_b)

    shared_cores_index = []
    shared_cores_index.append([])
    shared_cores_index.append([])

    for i in range(len(task_set)):
        # check if the task only require one type of processor
        # only require processor A:
        if task_set[i][1].utilizationB == 0 and task_set[i][1].utilizationA > 1:
            if available_a > 0:
                used_a = only_processor_a(task_set[i], available_a)
            else:
                return False, affinities, [-1, 0]
            if used_a > 0:
                affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [used_a, 0])
                available_a = available_a - used_a
            else:
                return False, affinities, [-1, 0]
        # only require processor B:
        elif task_set[i][1].utilizationA == 0 and task_set[i][1].utilizationB > 1:
            if available_b > 0:
                used_b = only_processor_b(task_set[i], available_b)
            else:
                return False, affinities, [0, -1]
            if used_b > 0:
                affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_task(current_index, [0, used_b])
                available_b = available_b - used_b
            else:
                return False, affinities, [0, -1]
        # total utilization no larger than 1
        elif task_set[i][0].utilization <= 1:
            # Initialize the shared processors
            if len(partition_ab[0]) == 0 and task_set[i][1].utilizationA > 0:
                partition_ab[0].append([])
                shared_cores_index[0].append(current_index[0])
                current_index[0] += 1
                available_a -= 1
            if len(partition_ab[1]) == 0 and task_set[i][1].utilizationB > 0:
                partition_ab[1].append([])
                shared_cores_index[1].append(current_index[1])
                current_index[1] += 1
                available_b -= 1

            temp_partition_ab = sched_share_light(task_set[i], partition_ab)
            if temp_partition_ab[0]:
                partition_ab = temp_partition_ab[1]
                if task_set[i][1].utilizationA > 0 and task_set[i][1].utilizationB > 0:
                    used_shared_index = [shared_cores_index[0][temp_partition_ab[2][0]], shared_cores_index[1][temp_partition_ab[2][1]]]
                elif task_set[i][1].utilizationA == 0:
                    used_shared_index = [-1,
                                         shared_cores_index[1][temp_partition_ab[2][1]]]
                else:
                    used_shared_index = [shared_cores_index[0][temp_partition_ab[2][0]], -1]
                affinities[task_set[i][0].tsk_id] = misc.assign_affinity_light_task(current_index, used_shared_index)[0]
                # the new task cannot be assigned on the used a/b processors
            else:
                temp_partition_ab = shared_light_newcore(task_set[i], partition_ab,
                                                         [available_a, available_b])
                if temp_partition_ab[0]:
                    # A new a core is added to the shared core cluster
                    if temp_partition_ab[2][0] > 0:
                        shared_cores_index[0].append(current_index[0])
                        current_index[0] += 1
                    # A new b core is added to the shared core cluster
                    if temp_partition_ab[2][1] > 0:
                        shared_cores_index[1].append(current_index[1])
                        current_index[1] += 1

                    if temp_partition_ab[3][0] < 0:
                        affinities[task_set[i][0].tsk_id] = misc.assign_affinity_light_task(current_index, [shared_cores_index[0][-1], shared_cores_index[1][-1]])[0]
                    else:
                        affinities[task_set[i][0].tsk_id] = misc.assign_affinity_light_task(current_index, [
                            shared_cores_index[0][temp_partition_ab[3][0]], shared_cores_index[1][temp_partition_ab[3][1]]])[0]
                    partition_ab = copy.deepcopy(temp_partition_ab[1])
                    available_a -= temp_partition_ab[2][0]
                    available_b -= temp_partition_ab[2][1]
                else:
                    return False, affinities, temp_partition_ab[2]

        # heavy_ab
        elif task_set[i][1].utilizationA > rho and task_set[i][1].utilizationB > rho:
            heavy_ab.append(task_set[i])

        # heavy_a
        elif task_set[i][1].utilizationA > rho >= task_set[i][1].utilizationB:
            temp_heavy_a = minimal_heavy_a(task_set[i], partitioned_b, rho)
            if temp_heavy_a and task_set[i][1].cpA < task_set[i][0].period / 3:
                if temp_heavy_a[1][1] > available_a:
                    heavy_ab.append(task_set[i])
                else:
                    partitioned_b = copy.deepcopy(temp_heavy_a[0])
                    available_a = available_a - temp_heavy_a[1][1]
                    affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_mix_heavy_task(current_index, [temp_heavy_a[1][1], shared_cores_index[1][temp_heavy_a[1][1]]], 0)
            else:
                temp_heavy_a = exclusive_heavy_a(task_set[i])
                if temp_heavy_a:
                    if temp_heavy_a > available_a or temp_heavy_a <= 0:
                        heavy_ab.append(task_set[i])
                    else:
                        new_core = []
                        new_task = []
                        new_task.append(task_set[i])
                        new_task.append(task_set[i][1].utilizationB * task_set[i][0].period + suspension_b(task_set[i], temp_heavy_a))
                        new_core.append(new_task)
                        partitioned_b.append(new_core)
                        shared_cores_index[1].append(current_index[1])
                        current_index[1] += 1
                        available_b -= 1
                        available_a = available_a - temp_heavy_a
                        affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_mix_heavy_task(
                            current_index, [temp_heavy_a, shared_cores_index[1][-1]], 0)
                else:
                    heavy_ab.append(task_set[i])

        # heavy_b
        elif task_set[i][1].utilizationA <= rho < task_set[i][1].utilizationB:
            temp_heavy_b = minimal_heavy_b(task_set[i], partitioned_a, rho)
            if temp_heavy_b and task_set[i][1].cpB < task_set[i][0].period / 3:
                if temp_heavy_b[1][1] > available_b:
                    heavy_ab.append(task_set[i])
                else:
                    affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_mix_heavy_task(
                        current_index, [shared_cores_index[0][temp_heavy_b[1][0]], temp_heavy_b[1][1]], 1)
                    partitioned_a = copy.deepcopy(temp_heavy_b[0])
                    available_b = available_b - temp_heavy_b[1][1]
            else:
                temp_heavy_b = exclusive_heavy_b(task_set[i])
                if temp_heavy_b:
                    if temp_heavy_b > available_b or temp_heavy_b <= 0:
                        heavy_ab.append(task_set[i])
                    else:
                        new_core = []
                        new_task = []
                        new_task.append(task_set[i])
                        new_task.append(task_set[i][1].utilizationA * task_set[i][0].period + suspension_a(task_set[i], temp_heavy_b))
                        new_core.append(new_task)
                        partitioned_a.append(new_core)
                        available_a -= 1
                        available_b = available_b - temp_heavy_b
                        shared_cores_index[0].append(current_index[0])
                        current_index[0] += 1
                        affinities[task_set[i][0].tsk_id], current_index = misc.assign_affinity_mix_heavy_task(
                            current_index, [shared_cores_index[0][-1], temp_heavy_b], 1)
                else:
                    heavy_ab.append(task_set[i])

        else:
            print("Some tasks are not considered!", task_set[i][0].utilization, task_set[i][1].utilizationA + task_set[i][1].utilizationB)

    # handle the heavy_ab tasks
    if len(heavy_ab) == 0:
        return True, affinities, current_index
    processor_ab = []
    lb_a_sum = 0
    lb_b_sum = 0
    for i in range(len(heavy_ab)):
        processor_ab_single = []
        lb_a = math.ceil(heavy_ab[i][1].utilizationA)
        lb_a_sum = lb_a_sum + lb_a
        lb_b = math.ceil(heavy_ab[i][1].utilizationB)
        lb_b_sum = lb_b_sum + lb_b

        schedulable_single_ab = False
        if lb_a_sum > available_a or lb_b_sum > available_b:
            return False, affinities, [available_a - lb_a_sum - len(heavy_ab) + i + 1, available_b - lb_b_sum - len(heavy_ab) + i + 1]
        else:
            for a in range(lb_a, available_a + 1):
                for b in range(lb_b, available_b + 1):
                    if han.hgsh(heavy_ab[i][0], heavy_ab[i][1], [a, b]) <= heavy_ab[i][0].period:
                        schedulable_single_ab = True
                        processor_ab_single.append([a, b])
                        break
        if schedulable_single_ab:
            processor_ab.append(processor_ab_single)
        else:
            return False, affinities, [-1, -1]

    # validate the possible combinations
    if len(processor_ab) <=0 and len(heavy_ab) > 0:
        return False, affinities, [-1, -1]

    feasible_ab = validate_heavy_ab_dp(processor_ab, available_a, available_b)
    # feasiblely allocate all the heavy ab tasks
    if feasible_ab:
        for ab in range(len(heavy_ab)):
            affinities[heavy_ab[ab][0].tsk_id], current_index = misc.assign_affinity_task(current_index, feasible_ab[ab])
        return True, affinities, current_index
    else:
        return False, affinities, [-1, -1]


def sufficient_greedy(task_set_org, typed_org, processor_a, processor_b, rho):
    task_set = copy.deepcopy(task_set_org)
    typed_info = copy.deepcopy(typed_org)
    sum_ua = 0
    sum_ub = 0

    for i in range(len(task_set)):
        if task_set[i].cp > (task_set[i].period*rho):
            return 0
        if max(typed_info[i].cpA, typed_info[i].cpB) > (task_set[i].period * rho):
            return 0
        else:
            sum_ua = sum_ua + typed_info[i].utilizationA
            sum_ub = sum_ub + typed_info[i].utilizationB
            if max(sum_ua/processor_a, sum_ub/processor_b) > rho:
                return 0
    return 1
