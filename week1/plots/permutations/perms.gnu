set terminal pngcairo
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "All algorithms with -O3 -ffast-math -floop-unroll"

plot 'mm_batch_11940762_mkn.out' title 'MKN', \
     'mm_batch_11940759_kmn.out' title 'KMN', \
     'mm_batch_11940766_nmk.out' title 'NMK', \
     'mm_batch_11940763_mnk.out' title 'MNK', \
     'mm_batch_11940760_knm.out' title 'KNM', \
     'mm_batch_11940765_nkm.out' title 'NKM', \
     'mm_batch_11978938_BLK_512.out' title 'BLK512', \
     'mm_batch_11943938_nat_blk_trans.out' title 'NAT'
