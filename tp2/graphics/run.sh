#! /bin/bash

comps=("gcc")
ns=("128" "256")

for comp in "${comps[@]}"; do
    for n in "${ns[@]}"; do
        python3 graphic_n.py ../sv "$comp" "$n"
        python3 graphic_flops.py ../sv "$comp" "$n"
    done
done