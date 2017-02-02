#!/bin/bash

#$ -S /bin/bash
#$ -cwd

params=( "$@" )
/data/icecube/software/LVTools_package/anaconda/envs/lvtools/bin/python /data/icecube/software/LVTools_package/LVTools/python/run.py -p ${params[0]} --min ${params[1]} --max ${params[2]} --n-points ${params[3]} --run ${params[4]}
