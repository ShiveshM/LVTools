#!/bin/bash

#$ -S /bin/bash
#$ -cwd

params=( "$@" )
/data/icecube/software/LVTools_package/anaconda/envs/lvtools/bin/python /data/icecube/software/LVTools_package/LVTools/python/run_spherical.py --this ${params[0]} --run ${params[1]} # --real " ${params[1]}"
