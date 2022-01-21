set terminal pngcairo size 800,380
set output "multigpu.png"

set xrange [0:1400]
set yrange [0:220]

set xlabel 'N'
set ylabel 'Performance [Gflops/s]'
#set title "blk runs with -O3 -ffast-math -floop-unroll"
set key outside

plot  'par.dat' using 1:($4/1000) title 'Single GPU', \
     'par2.dat' using 1:($4/1000) title 'Two GPUs'
