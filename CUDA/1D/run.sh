#! /bin/bash

make clean
clear
make
./noise_remover -i images/coffee.pgm -iter 10 -l 1 -o outputfile.pgm