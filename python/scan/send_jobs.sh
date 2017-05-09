#!/bin/bash
unset module;

min_range=-46;
max_range=-33;
N_POINTS=100;

increment=$(echo "($max_range - $min_range)/($N_POINTS - 1)" | bc -l);

run_index_asdf=1;
for a in $( seq $min_range $increment $max_range ); do
    echo $run_index_asdf
    qsub -cwd -V /data/icecube/software/LVTools_package/LVTools/python/scan/w.sh $a $min_range $max_range $N_POINTS $run_index_asdf;
    # /data/icecube/software/LVTools_package/LVTools/python/scan/w.sh $a $min_range $max_range $N_POINTS $run_index_asdf;
    run_index_asdf=$(echo "$run_index_asdf + 1" | bc -l);
done;
