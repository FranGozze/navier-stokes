#! /bin/bash

comps=("clang" "gcc" "icc" "icx")
ns=("128" "256" "512")

for comp in "${comps[@]}"; do
    for n in "${ns[@]}"; do
        python3 graphic_n.py ../sv "$comp" "$n"
        python3 graphic_var.py ../sv "$comp" "$n"
        python3 graphic_flops.py ../sv "$comp" "$n"
    done
done