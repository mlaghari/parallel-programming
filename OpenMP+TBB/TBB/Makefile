CC = icpc
CFLAGS  = -std=c++0x -g -Wall -O3 -tbb

default: imageSegmentation


imageSegmentation:  imseg_serial.o 
	$(CC) $(CFLAGS) -o imageSegmentation imseg_serial.o 

imseg_serial.o:  imseg_serial.cpp stb_image.h stb_image_write.h 
	$(CC) $(CFLAGS) -c imseg_serial.cpp


clean:
	$(RM) imageSegmentation *.o *~
