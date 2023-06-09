#!/bin/bash

echo "Archive the files from previous round ..."

destination_folder="../archive"
inputs_folder="../experiments/inputs/"
task_pure_folder="../experiments/inputs/tasks_pure"
task_type_folder="../experiments/inputs/tasks_typed"
task_data_folder="../experiments/inputs/tasks_data_request"

outputs_folder="../experiments/outputs/"
aff_folder="../experiments/outputs/affinity_allocation"
sched_folder="../experiments/outputs/schedule"

mkdir -p "$destination_folder"
folder_name=$(date +'%Y%m%d_%H%M%S')
destination_path="$destination_folder/$folder_name"
mkdir "$destination_path"

mv "$inputs_folder" "$destination_path"
mv "$outputs_folder" "$destination_path"
mv *.json "$destination_path"

if [ $? -eq 0 ]; then
  echo "Inputs and outputs folders moved successfully."
  echo "Destination: $destination_path"
else
  echo "Failed to move the inputs and outputs folders."
fi

echo "Create inputs and outputs folders"
mkdir -p "$task_pure_folder"
mkdir -p "$task_type_folder"
mkdir -p "$task_data_folder"
mkdir -p "$aff_folder"
mkdir -p "$sched_folder"

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