set terminal pngcairo
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "Permutation runs with -O3 -ffast-math"

plot 'mm_batch_11970703_mkn.out' title 'MKN', \
     'mm_batch_11970699_kmn.out' title 'KMN', \
     'mm_batch_11970707_nmk.out' title 'NMK', \
     'mm_batch_11970704_mnk.out' title 'MNK', \
     'mm_batch_11970701_knm.out' title 'KNM', \
     'mm_batch_11970706_nkm.out' title 'NKM'
