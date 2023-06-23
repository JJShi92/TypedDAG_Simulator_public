from collections import OrderedDict
from collections import defaultdict
from itertools import islice
import random
import copy
import sys
sys.path.append('../')
from algorithms import sched_sim
from generators import data_requests
from generators import generator_pure_dict
from generators import typed_core_allocation

# Cache defined by the ordered dictionary, the maximum size, and the access time.
# LRU is applied when the cache is full.
# Different layers of cache may have different access times.
class Cache:
    def __init__(self, max_size, access_time):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.access_time = access_time

    def get(self, address):
        if address in self.cache:
            # Move accessed address to the end to update its LRU status
            self.cache.move_to_end(address)
            return self.cache[address]
        else:
            return None

    def set(self, address, data):
        if address in self.cache:
            # Move updated address to the end to update its LRU status
            self.cache.move_to_end(address)
        else:
            if len(self.cache) >= self.max_size:
                # Remove the least recently used address (i.e., the first address) if cache is full
                self.cache.popitem(last=False)
        self.cache[address] = data


# We assume main memory contains all the needed data.
class Memory:
    def __init__(self, max_size, access_time):
        self.memory = OrderedDict()
        self.access_time = access_time
        self.max_size = max_size

    def get(self, address):
        # Access data from main memory
        return self.memory[address]

    def set(self, address, data):
        # Update main memory with the new data
        if address in self.memory:
            # Move updated address to the end to update its LRU status
            self.memory.move_to_end(address)
        else:
            if len(self.memory) >= self.max_size:
                # Remove the least recently used address (i.e., the first address) if memory is full
                self.memory.popitem(last=False)
        self.memory[address] = data

    def invalidate(self, address):
        # Invalidate data in main memory for the given address
        if address in self.memory:
            self.memory.pop(address)


# A modified three layers hierarchy is constructed for simulating the memory
# i.e., fast cache with main memory along with an independent fast memory.
# Different layers have different access times.
# Frequently accessed data will be stored in the fast memory with special address.
# The data in the fast memory has been determined in advance and statically, i.e., no update needed
# Other data will be stored in the two layers traditional memory with fast cache and main memory.
# A request will follow the following order:
# If the address points to the fast memory, go to fast memory directly,
# Otherwise, check the fast cache and main memory, until the data is found.
# Once the data is found in the main memory, a synchronization process will be operated.
# In schedule simulation, only tne access time is returned, data is assumed to be correctly returned by default.
class ThreeLayerCache:
    def __init__(self, l1_size, access_time_1, main_memory, fast_memory):
        self.l1_cache = Cache(l1_size, access_time_1)
        self.main_memory = main_memory
        self.fast_memory = fast_memory

    def get(self, address):
        real_access_time = 0
        # Check the fast_memory at first
        if address in self.fast_memory.memory:
            # print("Data found in fast memory.")
            real_access_time = self.fast_memory.access_time
        else:
            # Check L1 cache afterwards
            if address in self.l1_cache.cache:
                # print("Data found in L1 cache.")
                real_access_time = self.l1_cache.access_time
            else:
                # Access data from main memory and update L1 caches
                data = self.main_memory.get(address)
                if data is not None:
                    # print("Data found in main memory.")
                    # Update L1 cache
                    self.l1_cache.set(address, data)
                    real_access_time = self.main_memory.access_time
                else:
                    # Even maim memory does not have the data, something must be wrong
                    print("Something is wrong!")
        return real_access_time

    # Only set the main memory with the new data
    def set_main_single(self, address, data):
        self.main_memory.set(address, data)
        # print("Add data to main memory.")

    def set_main_both(self, address, data):
        # Update l1 cache and main memory with the new data
        self.l1_cache.set(address, data)
        self.main_memory.set(address, data)
        # print("Add data to L1 cache, and main memory.")

    def set_fastmemory(self, address, data):
        # Update fast memory with the new data
        self.fast_memory.set(address, data)
        # print("Add data to fast memory.")

    def invalidate(self, address):
        # Invalidate data in all layers of cache and main memory for the given address
        if address in self.l1_cache.cache:
            self.l1_cache.cache.pop(address)
        if address in self.fast_memory.memory:
            self.fast_memory.invalidate(address)
        if address in self.main_memory.memory:
            self.main_memory.invalidate(address)
        print("Data invalidated in fast memory, L1 caches, and main memory for the given address")


# Count the requested data
# find the frequently requested data
# periods are taken into consideration
# help for the initialization
def count_requested_data(requested_data_set_org, hp, periods):
    requested_data_set = copy.deepcopy(requested_data_set_org)
    # Initialize a dictionary to store the counts
    counts = defaultdict(float)

    i = 0
    for data_task in requested_data_set:
        for key, value in enumerate(data_task.req_data.items()):
            if value[0] != 0 and value[0] != data_task.V-1:
                for it in range(len(value[1])):
                    counts[value[1][it]] += int(hp/periods[i]) * data_task.req_prob[value[0]][it]
        i += 1
    # sort the counts by their values decreasingly
    sorted_counts_values = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    return(sorted_counts_values)


# Initialize the memory configuration
# mod0: only initialize the fast memory and main memory
# mod1: initialize the fast memory, l1 cache, and main memory together
def memory_initialization(l1_size, access_time_l1, fast_mem_size, access_time_fast, main_mem_size, access_time_main, requested_data_set_org, hp, periods, mod):
    requested_data_set = copy.deepcopy(requested_data_set_org)
    # generate the memory hierarchy
    Main_memory = Memory(main_mem_size, access_time_main)
    Fast_memory = Memory(fast_mem_size, access_time_fast)
    three_layer_cache = ThreeLayerCache(l1_size, access_time_l1, Main_memory, Fast_memory)

    # record the statistics of requested data
    counted_data = count_requested_data(requested_data_set, hp, periods)
    counted_data_list = list(counted_data.keys())

    for i in range(len(counted_data)):
        if i < fast_mem_size:
            three_layer_cache.set_fastmemory(counted_data_list[i], "Data")
        else:
            if mod == 0:
                three_layer_cache.l1_cache.cache.clear()
                three_layer_cache.set_main_single(counted_data_list[i], "Data")
            else:
                if i < fast_mem_size + l1_size:
                    three_layer_cache.set_main_both(counted_data_list[i], "Data")
                else:
                    three_layer_cache.set_main_single(counted_data_list[i], "Data")

    return three_layer_cache


'''
t0 = []
for i in range(10):
    t1 = defaultdict(hex)
    for j in range(8):
        t1[j] = hex(random.randint(990, 1000))
    t0.append(t1)

key_list = list(count_requested_data(t0).keys())

three_mem = memory_initialization(2, 1, 3, 10, 10, 100, t0, 0)
three_mem_2 = memory_initialization(2, 1, 3, 10, 10, 100, t0, 1)
print("try different mod")
print("mod 0:", three_mem.get(key_list[1]))
print("mod 1:", three_mem_2.get(key_list[1]))
print("mod 0:", three_mem.get(key_list[3]))
print("mod 1:", three_mem_2.get(key_list[3]))
print("mod 0:", three_mem.get(key_list[-1]))
print("mod 1:", three_mem_2.get(key_list[-1]))

print("End")


for i in range(8):
    print(three_mem.get(t0[0][i]))
    print("mod 1", three_mem_2.get(t0[0][i]))


# Create main memory
main_memory = Memory(10000, 100)
fast_memory = Memory(100, 10)

# Create a three-layer cache with L1 size of 4, L2 size of 8, and main memory
three_layer_cache = ThreeLayerCache(4, 1, main_memory, fast_memory)


my_keys = c.keys()
for key in my_keys:
    print(key)
    main_memory.set(key, "Data")

my_keys_list = list(my_keys)
print(three_layer_cache.get(my_keys_list[2]))



# Set initial data in main memory
main_memory.set(0, "Data0")
main_memory.set(0x100, "Data1")
main_memory.set(0x200, "Data2")
main_memory.set(0x300, "Data3")
main_memory.set(0x400, "Data4")
main_memory.set(0xAD00, "Data4")
main_memory.set(hex(random.randint(1, 200)), "Datax")

fast_memory.set(0x2100, "Data5")
fast_memory.set(0x2200, "Data6")


# Access data from the three-layer cache
print(three_layer_cache.get(0))
print(three_layer_cache.get(0x100))  # Data1 (not found in caches, retrieved from main memory and added to L1 and L2 caches)
print(three_layer_cache.get(0x200))  # Data2 (not found in caches, retrieved from main memory and added to L1 and L2 caches)

#print(three_layer_cache.l2_cache.set(0x300, "Data3"))
print(three_layer_cache.get(0x300))

print(three_layer_cache.get(0x2100))
print(three_layer_cache.get(0x2200))
# Access data from the three-layer cache again
print(three_layer_cache.get(0x100))  # Data1 (found in L1 cache, moved to end of LRU queue)
print(three_layer_cache.get(0x200))  # Data2 (found in L1 cache, moved to end of LRU queue)

# Set new data in the three-layer cache
three_layer_cache.set_main(0x500, "Data5")  # Data5 (added to L1, L2 caches, and main memory)

# Access data from the three-layer cache after setting new data
print(three_layer_cache.get(0x100))  # Data1 (found in L1 cache, moved to end of LRU queue)
print(three_layer_cache.get(0x500))  # Data5 (found in L1 cache, moved to end of LRU queue)
'''