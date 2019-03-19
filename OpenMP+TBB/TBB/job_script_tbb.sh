#!/bin/bash
#$ -N ml_job_tbb
#$ -S /bin/bash
#$ -q short.q
#$ -pe smp 16
#$ -o $JOB_ID.out
#$ -e $JOB_ID.err
#$ -M mlaghari16@ku.edu.tr
#$ -m bea


echo $HOSTNAME
echo "Image segmentation job script"


echo "-------------COFFEE-------------"
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 32
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 16
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 8
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 4
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 2

echo "-------------WINDOWS-------------"
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 32
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 16
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 8
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 4
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 2

echo "-------------HANGER-------------"
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 32
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 16
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 8
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 4
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 2

