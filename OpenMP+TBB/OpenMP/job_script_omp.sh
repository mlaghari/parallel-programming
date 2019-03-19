#!/bin/bash
#$ -N ml_job
#$ -S /bin/bash
#$ -q short.q
#$ -pe smp 16
#$ -o $JOB_ID.out
#$ -e $JOB_ID.err
#$ -M mlaghari16@ku.edu.tr
#$ -m bea


echo $HOSTNAME
echo "Image segmentation job script"

echo "-------------Coffee-------------"
# ./imageSegmentation_serial -i ./coffee.jpg -s 2 -o r_coffee_serial.png
export OMP_NUM_THREADS=32
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_32.png
export OMP_NUM_THREADS=16
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_16.png
export OMP_NUM_THREADS=8
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_8.png
export OMP_NUM_THREADS=4
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_4.png
export OMP_NUM_THREADS=2
./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_2.png

echo "-------------Windows-------------"
# ./imageSegmentation_serial -i ./windows.jpg -s 2 -o r_windows_serial.png
export OMP_NUM_THREADS=32
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_32.png
export OMP_NUM_THREADS=16
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_16.png
export OMP_NUM_THREADS=8
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_8.png
export OMP_NUM_THREADS=4
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_4.png
export OMP_NUM_THREADS=2
./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_2.png

echo "-------------Hanger-------------"
# ./imageSegmentation_serial -i ./hanger.jpg -s 2 -o r_hanger_serial.png
export OMP_NUM_THREADS=32
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_32.png
export OMP_NUM_THREADS=16
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_16.png
export OMP_NUM_THREADS=8
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_8.png
export OMP_NUM_THREADS=4
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_4.png
export OMP_NUM_THREADS=2
./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_2.png

# ./imageSegmentation -i ./coffee.jpg -s 2 -o r_coffee_tbb.png -t 32
# ./imageSegmentation -i ./windows.jpg -s 2 -o r_windows_tbb.png -t 32
# ./imageSegmentation -i ./hanger.jpg -s 2 -o r_hanger_tbb.png -t 32
