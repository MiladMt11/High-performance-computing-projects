set terminal pngcairo
set output "flagcompare.png"

set logscale x
set yrange [0:3500]
set key bottom right

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "MKN runs with different flags"

plot    'o2_mkn.out' title '-O2', \
        'o3_all.out' title '-O3 -ffast-math -floop-unroll', \
     'o3_fm_mkn.out' title '-O3 -ffast-math', \
     'o3_lu_mkn.out' title '-O3 -floop-unroll', \
        'o3_mkn.out' title '-O3', \
     'no_op_mkn.out' title 'No flags'
