set terminal pngcairo
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "Permutation runs with -O3"

plot 'mm_batch_11970574_mkn.out' title 'MKN', \
     'mm_batch_11970571_kmn.out' title 'KMN', \
     'mm_batch_11970578_nmk.out' title 'NMK', \
     'mm_batch_11970575_mnk.out' title 'MNK', \
     'mm_batch_11970572_knm.out' title 'KNM', \
     'mm_batch_11970577_nkm.out' title 'NKM'
