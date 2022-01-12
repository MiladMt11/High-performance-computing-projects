set output "seq_updates_per_second.png"
set terminal pngcairo size 800,400

set xlabel "N"
set ylabel "Million lattice site updates per second"

set key top right

set xrange [19:151]
set yrange [0:350]

plot 'seq_updates_per_sec_j.out' using 1:4 title 'Jacobi', \
     'seq_updates_per_sec_gs.out' using 1:4 title 'Gauss Seidel'
