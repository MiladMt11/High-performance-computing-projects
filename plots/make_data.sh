#!/bin/sh

mkdir -p $1
cp $2/*.out $1

for f in $1/*.out; do sed -i '/ # matmult_/!d' $f; done
