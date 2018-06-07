#!/bin/bash
FORMULATION=$1
GP=$2
shift 2

PENALTY=1.
RUNDIR="$HOME/m/nmlog/$FORMULATION-$GP$PENALTY"

python gabriel_main.py $RUNDIR --gp $PENALTY --formulation $FORMULATION --gp-type $GP --dataset cifar10 --dataroot cifar --log-every 10 $@  # get remaining args