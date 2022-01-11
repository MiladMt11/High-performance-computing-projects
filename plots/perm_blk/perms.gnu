set terminal pngcairo size 820,480
set output "perms.png"

set logscale x

set xlabel 'Memory footprint [Kb]'
set ylabel 'Performance [Mflops/s]'
set title "blk runs with -O3 -ffast-math -floop-unroll"
set key outside

plot     'mm_batch_11978940_BLK_8.out' title 'BLK8', \
        'mm_batch_11978939_BLK_64.out' title 'BLK64', \
       'mm_batch_11978932_BLK_128.out' title 'BLK128', \
       'mm_batch_11978935_BLK_256.out' title 'BLK256', \
       'mm_batch_11978938_BLK_512.out' title 'BLK512', \
      'mm_batch_11978931_BLK_1024.out' title 'BLK1024', \
      'mm_batch_11978934_BLK_2048.out' title 'BLK2048', \
      'mm_batch_11978937_BLK_4096.out' title 'BLK4096', \
      'mm_batch_11979384_BLK_4100.out' title 'BLK4100', \
      'mm_batch_11978941_BLK_8192.out' title 'BLK8192', \
     'mm_batch_11978933_BLK_16384.out' title 'BLK16384', \
     'mm_batch_11978936_BLK_32768.out' title 'BLK32768'
