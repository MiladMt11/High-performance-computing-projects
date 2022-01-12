set output "iters_to_convergence.png"
set terminal pngcairo

set xlabel "N"
set ylabel "Iterations to convergence"

set key top left

plot 'iters_to_convergence_j.out' title 'Jacobi', \
     'iters_to_convergence_gs.out' title 'Gauss Seidel'
