from collections import defaultdict


# Generate the affinities for each task without any specific algorithm
# Each vertex can be executed on any processors with the same type
def gen_affinities_raw(task_set_org, processor_a, processor_b):
    affinities = defaultdict(list)

    # Generate the processor lists:
    processor_a_list = list(range(0, processor_a))
    processor_b_list = list(range(processor_a, processor_a + processor_b))

    # Append the affinity for each task
    for i in range(len(task_set_org)):
        affinity = defaultdict(list)
        affinity[1] = processor_a_list
        affinity[2] = processor_b_list

        affinities[i] = affinity

    return affinities