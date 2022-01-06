#!/bin/sh

cd perm_o2 && gnuplot perms.gnu && cd ..
cd perm_o3 && gnuplot perms.gnu && cd ..
cd perm_o3_fm && gnuplot perms.gnu && cd ..
cd perm_o3_lu && gnuplot perms.gnu && cd ..
cd permutations && gnuplot perms.gnu && cd ..
