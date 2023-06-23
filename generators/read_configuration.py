from __future__ import division
import os
import math
import sys
import getopt
import json


def read_conf(conf_file_name):

    with open(conf_file_name, 'r') as json_file:
        data = json.load(json_file)

    conf_names = list(data.keys())

    # Initialize a dictionary to store attribute settings
    conf_settings = {name: set() for name in conf_names}

    # Extract the settings for each conf
    for name in conf_names:
        values = data[name]
        conf_settings[name].update(values)

    # Sort the values in each setting
    conf_settings = {key: sorted(values) for key, values in conf_settings.items()}

    return conf_settings
