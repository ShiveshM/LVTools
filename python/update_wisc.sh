#!/bin/bash

set -e

while sleep 5; do
    rsync -avut /data/icecube/software/LVTools_package/LVTools/python/nohup.out -e 'ssh -X' smandalia@data.icecube.wisc.edu:/home/smandalia/lv/progress.out
done
