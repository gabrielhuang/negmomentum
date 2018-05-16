#!/bin/bash
FORMULATION=$1
GP=$2
shift 2
RUNDIR="$FORMULATION-$GP"

python gabriel_main.py "log/$RUNDIR" --formulation $FORMULATION --gp-type $GP --dataset cifar10 --dataroot cifar --log-every 10 $@  # get remaining args
