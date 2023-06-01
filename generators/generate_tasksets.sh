#!/bin/bash

echo "Task sets generation starts ..."

python3 configuration_generator.py
sleep 1

python3 tasksets_generator_pure.py
sleep 1

python3 tasksets_generator_typed.py
sleep 1

python3 tasksets_generator_data_requests.py
sleep 1

echo "Taks sets have been successfully generated!"