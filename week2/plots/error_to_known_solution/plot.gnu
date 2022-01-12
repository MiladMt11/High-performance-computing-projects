set output "error_to_known_solution.png"
set terminal pngcairo size 800,400

set xlabel "N"
set ylabel "Error"

set key top right

set xrange [9:101]
set yrange [0:0.4]

plot 'error.dat' title 'Error'
