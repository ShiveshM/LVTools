#!/bin/bash
unset module;

set -e

this="a";
n=100;

python gen_cart_from_sph.py --this $this --runs $n
run_idx=1;
# for re in `echo -e 'import mpmath as mp\nmp.mp.dps = 50\nimport cPickle as pickle\n\nimport numpy as np\na = pickle.load(open("./numpy/'$this'.pckl", "r"))\nfor x in np.unique(a[0]):\n\tprint x' | python`; do
for ((i=0; i<n; i++)); do
    echo $run_idx;
    # /data/icecube/software/LVTools_package/LVTools/python/scan/w_spherical.sh $this $run_idx;
    qsub -cwd -V /data/icecube/software/LVTools_package/LVTools/python/scan/w_spherical.sh $this $run_idx;
    run_idx=$(echo "$run_idx + 1" | bc -l);
done;
