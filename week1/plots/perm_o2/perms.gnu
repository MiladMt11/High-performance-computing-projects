set terminal pngcairo
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "Permutation runs with -O2"

plot 'mm_batch_11970537_mkn.out' title 'MKN', \
     'mm_batch_11970533_kmn.out' title 'KMN', \
     'mm_batch_11970541_nmk.out' title 'NMK', \
     'mm_batch_11970538_mnk.out' title 'MNK', \
     'mm_batch_11970534_knm.out' title 'KNM', \
     'mm_batch_11970540_nkm.out' title 'NKM'
