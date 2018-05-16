#!/bin/bash
RUNDIR="dcgan"
shift 0

python gabriel_main.py "log/$RUNDIR" --dataset cifar10 --dataroot cifar $@  # get remaining args
