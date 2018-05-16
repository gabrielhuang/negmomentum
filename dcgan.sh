#!/bin/bash
FORMULATION=$1
GP=$2
shift 2
RUNDIR="$HOME/m/nmlog/$FORMULATION-$GP"

python gabriel_main.py $RUNDIR --formulation $FORMULATION --gp-type $GP --dataset cifar10 --dataroot cifar --log-every 10 $@  # get remaining args
