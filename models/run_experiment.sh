#!/bin/bash

if [ -z "$1" ]
	then
		echo Experiment id required
		exit 1
fi

echo Running experiment "$1"
for MODEL in vae isvae lisvae
do
	python train.py --model=$MODEL --experiment_id=$1	
done
