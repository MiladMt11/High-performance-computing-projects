set terminal pngcairo
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "Permutation runs with -O3 -floop-unroll"

plot 'mm_batch_11971329_mkn.out' title 'MKN', \
     'mm_batch_11971326_kmn.out' title 'KMN', \
     'mm_batch_11971334_nmk.out' title 'NMK', \
     'mm_batch_11971330_mnk.out' title 'MNK', \
     'mm_batch_11971327_knm.out' title 'KNM', \
     'mm_batch_11971333_nkm.out' title 'NKM'
