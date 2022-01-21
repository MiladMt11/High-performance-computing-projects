set terminal pngcairo size 800,380
set output "speedup2.png"

set xrange [0:1400]
set yrange [0:240]

set xlabel 'N'
set ylabel 'Performance [Gflops/s]'
set key outside

plot 'cpu.dat' using 1:($4/1000) title '2 CPUs', \
     'gpu.dat' using 1:($4/1000) title '2 GPUs'
