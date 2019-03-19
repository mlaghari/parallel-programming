/*	
 * noise_remover.cpp
 *
 * This program removes noise from an image based on Speckle Reducing Anisotropic Diffusion
 * Y. Yu, S. Acton, Speckle reducing anisotropic diffusion, 
 * IEEE Transactions on Image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
 * Original implementation is Modified by Burak BASTEM and Nufail Farooqi
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <mpi.h>

using namespace MPI;

#define MASTER 0
#define SOUTH_TO_NORTH 1
#define NORTH_TO_SOUTH 2
#define FIRST_TO_LAST 3
#define LAST_TO_FIRST 4

#define MATCH(s) (!strcmp(argv[ac], (s)))

// returns the current time
static const double kMicro = 1.0e-6;
double get_time() {
	struct timeval TV;
	struct timezone TZ;
	const int RC = gettimeofday(&TV, &TZ);
	if(RC == -1) {
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}
	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

void update_image_ghostcells(unsigned char *image, int height, int width);
void update_coeff_ghostcells(float *coeff, int height, int width);

void update_image_ghostcells_extended(unsigned char *image, int height, int width);
void update_coeff_ghostcells_extended(float *coeff, int height, int width);

int main(int argc, char *argv[]) {
	// Part I: allocate and initialize variables
	double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
	time_0 = get_time();
	const char *filename = "input.pgm";
	const char *outputname = "output.png";	
	int width, height, pixelWidth;
	int heightPerProcess, widthPerProcess;
	long n_pixels,n_pixels_with_ghost;
	int n_iter = 50;
	float lambda = 0.5;
	float mean, variance, std_dev;	//local region statistics
	float *north_deriv, *south_deriv, *west_deriv, *east_deriv;	// directional derivatives
	double tmp, sum, sum2, temp_sum, temp_sum2;	// calculation variables
	float gradient_square, laplacian, num, den, std_dev2, divergence;	// calculation variables
	float *diff_coef;	// diffusion coefficient
	float *lcl_diff_coef;
	float diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east;	// directional diffusion coefficients
	long k, k2;	// current pixel index
	int px=0, py=0;
	int heightPerRank = 0;
    int rank=0, size=1; 
    Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank ); 
    MPI_Comm_size(MPI_COMM_WORLD, &size ); 
    // printf("Hello from process %d of %d\n", rank, size);
 
	time_1 = get_time();	
	
	// Part II: parse command line arguments
	if(argc<2) {
	  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>][-x <processor geometery in x>] [-y <processor geometery in y>]\n",argv[0]);
	  return(-1);
	}
	for(int ac=1;ac<argc;ac++) {
		if(MATCH("-i")) {
			filename = argv[++ac];
		} else if(MATCH("-iter")) {
			n_iter = atoi(argv[++ac]);
		} else if(MATCH("-x")) {
		  px = atoi(argv[++ac]);
		} else if(MATCH("-y")) {
		  py = atoi(argv[++ac]);	
		} else if(MATCH("-l")) {
			lambda = atof(argv[++ac]);
		} else if(MATCH("-o")) {
			outputname = argv[++ac];
		} else {
		  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>] [-x <processor geometery in x>] [-y <processor geometery in y>]\n",argv[0]);
		return(-1);
		}
	}
	time_2 = get_time();

	// Part III: read image	
	unsigned char *tmp_image;
	if (rank == MASTER) {
		printf("Reading image...\n");
		tmp_image = stbi_load(filename, &width, &height, &pixelWidth, 0);;
		if (!tmp_image) {
			fprintf(stderr, "Couldn't load image.\n");
			return (-1);
		}
		printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);	
		// printf("Rank: %d, Height: %d, Width: %d\n", rank, height, width);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&width, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	heightPerProcess = (height)/size;
	printf("hPP %d\n", heightPerProcess);
	unsigned char *image = (unsigned char*) malloc(sizeof(unsigned char) * ((height)*(width)));
	unsigned char *lclimage = (unsigned char*) malloc(sizeof(unsigned char) * ((heightPerProcess+2)*(width+2)));
	MPI_Scatter(tmp_image, heightPerProcess*(width), MPI_UNSIGNED_CHAR, image, heightPerProcess*(width), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	// printf("Here\n");
	MPI_Barrier(MPI_COMM_WORLD);
	n_pixels = height * width;
	n_pixels_with_ghost = (height+2) * (width+2);
	// Copy the image into extended image array(with ghost cells)
	for (int i = 1; i <= heightPerProcess ; i++) {
	  for (int j = 1; j <= width ; j++) {
	    k = i * (width+2) + j;	// position of current element
	    k2 = (i-1) * width + (j-1);
	    
	    lclimage[k] = image[k2];
	  }
	}
	time_3 = get_time();
	
	
	// Part IV: allocate variables
	north_deriv = (float*) malloc(sizeof(float) * n_pixels);	// north derivative
	south_deriv = (float*) malloc(sizeof(float) * n_pixels);	// south derivative
	west_deriv = (float*) malloc(sizeof(float) * n_pixels);	// west derivative
	east_deriv = (float*) malloc(sizeof(float) * n_pixels);		// east derivative
	diff_coef  = (float*) malloc(sizeof(float) * n_pixels_with_ghost);	// diffusion coefficient
	lcl_diff_coef  = (float*) malloc(sizeof(float) * ((heightPerProcess+2)*(width+2)));	// diffusion coefficient
	
	time_4 = get_time();
	
	// Part V: compute --- n_iter * (3 * height * width + 41 * (height-1) * (width-1) + 6) floating point arithmetic operations in totaL
	temp_sum = 0;
	temp_sum2 = 0;
	for (int iter = 0; iter < n_iter; iter++) {
		sum = 0;
		sum2 = 0;
		temp_sum = 0;
		temp_sum2 = 0;
		
		if (size > 1) {
			update_image_ghostcells(lclimage, heightPerProcess+2, width+2);
			// REDUCTION AND STATISTICS
			// --- 3 floating point arithmetic operations per element -> 3*height*width in total		
			for (int i = 1 ; i <= heightPerProcess ; i++) {
				for (int j = 1; j <= width; j++) {
				        tmp = lclimage[i * (width+2) + j];	// current pixel value
					temp_sum += tmp; // --- 1 floating point arithmetic operations
					temp_sum2 += tmp * tmp; // --- 2 floating point arithmetic operations
				}
			}
	    	MPI_Reduce(&temp_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
	    	MPI_Reduce(&temp_sum2, &sum2, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
	        MPI_Bcast(&sum, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	        MPI_Bcast(&sum2, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	        if (rank == MASTER) {
	        	// printf("Sum: %f, Sum2: %f\n", sum, sum2);
	        }

			mean = sum / n_pixels; // --- 1 floating point arithmetic operations
			variance = (sum2 / n_pixels) - mean * mean; // --- 3 floating point arithmetic operations
			std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
					
		} else {
			update_image_ghostcells_extended(lclimage, heightPerProcess+2, width+2);
			for (int i = 1; i <= height; i++) {
				for (int j = 1; j <= width; j++) {
				        tmp = image[i * (width+2) + j];	// current pixel value
					sum += tmp; // --- 1 floating point arithmetic operations
					sum2 += tmp * tmp; // --- 2 floating point arithmetic operations
				}
			}
			// printf("Sum: %f, Sum2: %f\n", sum, sum2);
			mean = sum / n_pixels; // --- 1 floating point arithmetic operations
			variance = (sum2 / n_pixels) - mean * mean; // --- 3 floating point arithmetic operations
			std_dev = variance / (mean * mean); // --- 2 floating point arithmetic operations
			
		}
		
		//COMPUTE 1
		// --- 32 floating point arithmetic operations per element -> 32*(height-1)*(width-1) in total
		if (size > 1) {
			if (rank != 0) {
				MPI_Send(lclimage+width+2, width+2, MPI_UNSIGNED_CHAR, rank-1, SOUTH_TO_NORTH, MPI_COMM_WORLD);
				MPI_Recv(lclimage, width+2, MPI_UNSIGNED_CHAR, rank-1, NORTH_TO_SOUTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			} 

			if (rank != size-1) {
				MPI_Recv(lclimage+((heightPerProcess + 1)*(width+2)), width+2, MPI_UNSIGNED_CHAR, rank+1, SOUTH_TO_NORTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(lclimage+((heightPerProcess)*(width+2)), width+2, MPI_UNSIGNED_CHAR, rank+1, NORTH_TO_SOUTH, MPI_COMM_WORLD);
			}

			if (rank == 0) {
				MPI_Send(lclimage+width+2, width+2, MPI_UNSIGNED_CHAR, size-1, FIRST_TO_LAST, MPI_COMM_WORLD);
				MPI_Recv(lclimage, width+2, MPI_UNSIGNED_CHAR, size-1, LAST_TO_FIRST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			if (rank == size-1) {
				MPI_Recv(lclimage+((heightPerProcess+1)*(width+2)), width+2, MPI_UNSIGNED_CHAR, MASTER, FIRST_TO_LAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(lclimage+((heightPerProcess)*(width+2)), width+2, MPI_UNSIGNED_CHAR, MASTER, LAST_TO_FIRST, MPI_COMM_WORLD);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			
			for (int i = 1; i <= heightPerProcess ; i++) {
				for (int j = 1; j <= width ; j++) {
					k2 = (i-1) * width + (j-1);	// position of current element
					k = i * (width+2) + j;
					
					north_deriv[k2] = lclimage[(i - 1) * (width+2) + j] - lclimage[k];	// north derivative --- 1 floating point arithmetic operations
					south_deriv[k2] = lclimage[(i + 1) * (width+2) + j] - lclimage[k];	// south derivative --- 1 floating point arithmetic operations
					west_deriv[k2] = lclimage[i * (width+2) + (j - 1)] - lclimage[k];	// west derivative --- 1 floating point arithmetic operations
					east_deriv[k2] = lclimage[i * (width+2) + (j + 1)] - lclimage[k];	// east derivative --- 1 floating point arithmetic operations

					gradient_square = (north_deriv[k2] * north_deriv[k2] + south_deriv[k2] * south_deriv[k2] + west_deriv[k2] * west_deriv[k2] + east_deriv[k2] * east_deriv[k2]) / (lclimage[k] * lclimage[k]); // 9 floating point arithmetic operations
					laplacian = (north_deriv[k2] + south_deriv[k2] + west_deriv[k2] + east_deriv[k2]) / lclimage[k]; // 4 floating point arithmetic operations
					num = (0.5 * gradient_square) - ((laplacian * laplacian)/16.0); // 4 floating point arithmetic operations
					den = 1 + (.25 * laplacian); // 2 floating point arithmetic operations
					std_dev2 = num / (den * den); // 2 floating point arithmetic operations
					den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 floating point arithmetic operations
					lcl_diff_coef[k] = 1.0 / (1.0 + den); // 2 floating point arithmetic operations
					if (lcl_diff_coef[k] < 0) {
						lcl_diff_coef[k] = 0;
					} else if (lcl_diff_coef[k] > 1)	{
						lcl_diff_coef[k] = 1;
					}
				}
			}
		} else {
			for (int i = 1; i <= height ; i++) {
				for (int j = 1; j <= width ; j++) {
					k2 = (i-1) * width + (j-1);	// position of current element
					k = i * (width+2) + j;
					north_deriv[k2] = lclimage[(i - 1) * (width+2) + j] - lclimage[k];	// north derivative --- 1 floating point arithmetic operations
					south_deriv[k2] = lclimage[(i + 1) * (width+2) + j] - lclimage[k];	// south derivative --- 1 floating point arithmetic operations
					west_deriv[k2] = lclimage[i * (width+2) + (j - 1)] - lclimage[k];	// west derivative --- 1 floating point arithmetic operations
					east_deriv[k2] = lclimage[i * (width+2) + (j + 1)] - lclimage[k];	// east derivative --- 1 floating point arithmetic operations

					gradient_square = (north_deriv[k2] * north_deriv[k2] + south_deriv[k2] * south_deriv[k2] + west_deriv[k2] * west_deriv[k2] + east_deriv[k2] * east_deriv[k2]) / (lclimage[k] * lclimage[k]); // 9 floating point arithmetic operations
					laplacian = (north_deriv[k2] + south_deriv[k2] + west_deriv[k2] + east_deriv[k2]) / image[k]; // 4 floating point arithmetic operations
					num = (0.5 * gradient_square) - ((laplacian * laplacian)/16.0); // 4 floating point arithmetic operations
					den = 1 + (.25 * laplacian); // 2 floating point arithmetic operations
					std_dev2 = num / (den * den); // 2 floating point arithmetic operations
					den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 floating point arithmetic operations
					lcl_diff_coef[k] = 1.0 / (1.0 + den); // 2 floating point arithmetic operations
					if (lcl_diff_coef[k] < 0) {
						lcl_diff_coef[k] = 0;
					} else if (lcl_diff_coef[k] > 1)	{
						lcl_diff_coef[k] = 1;
					}
				}
			}
		}

		if (size > 1) {
			update_coeff_ghostcells(lcl_diff_coef, heightPerProcess+2, width+2);
			if (rank != 0) {
				MPI_Send(lcl_diff_coef+width+2, width+2, MPI_FLOAT, rank-1, SOUTH_TO_NORTH, MPI_COMM_WORLD);
				MPI_Recv(lcl_diff_coef, width+2, MPI_FLOAT, rank-1, NORTH_TO_SOUTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			} 

			if (rank != size-1) {
				MPI_Recv(lcl_diff_coef+((heightPerProcess+1)*(width+2)), width+2, MPI_FLOAT, rank+1, SOUTH_TO_NORTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(lcl_diff_coef+((heightPerProcess)*(width+2)), width+2, MPI_FLOAT, rank+1, NORTH_TO_SOUTH, MPI_COMM_WORLD);
			}
			if (rank == 0) {
				MPI_Send(lcl_diff_coef+width+2, width+2, MPI_FLOAT, size-1, FIRST_TO_LAST, MPI_COMM_WORLD);
			}
			if (rank == size-1) {
				MPI_Recv(lcl_diff_coef+((heightPerProcess+1)*(width+2)), width+2, MPI_FLOAT, MASTER, FIRST_TO_LAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			for (int i = 1; i <= heightPerProcess; i++) {
				for (int j = 1; j <= width; j++) {
				  k2 = (i-1) * width + (j-1);
				  k = i * (width+2) + j;	// get position of current element
				  
				  diff_coef_north = lcl_diff_coef[k];	// north diffusion coefficient
				  diff_coef_south = lcl_diff_coef[(i + 1) * (width+2) + j];	// south diffusion coefficient
				  diff_coef_west = lcl_diff_coef[k];	// west diffusion coefficient
				  diff_coef_east = lcl_diff_coef[i * (width+2) + (j + 1)];	// east diffusion coefficient
				  
				  divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations
				  lclimage[k] = lclimage[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
				}
			}
		} else {
			update_coeff_ghostcells_extended(lcl_diff_coef, heightPerProcess+2, width+2);
			for (int i = 1; i <= height; i++) {
				for (int j = 1; j <= width; j++) {
				  k2 = (i-1) * width + (j-1);
				  k = i * (width+2) + j;	// get position of current element
				  
				  diff_coef_north = lcl_diff_coef[k];	// north diffusion coefficient
				  diff_coef_south = lcl_diff_coef[(i + 1) * (width+2) + j];	// south diffusion coefficient
				  diff_coef_west = lcl_diff_coef[k];	// west diffusion coefficient
				  diff_coef_east = lcl_diff_coef[i * (width+2) + (j + 1)];	// east diffusion coefficient
				  
				  divergence = diff_coef_north * north_deriv[k2] + diff_coef_south * south_deriv[k2] + diff_coef_west * west_deriv[k2] + diff_coef_east * east_deriv[k2]; // --- 7 floating point arithmetic operations
				  lclimage[k] = lclimage[k] + 0.25 * lambda * divergence; // --- 3 floating point arithmetic operations
				}
			}
		}
		
	}
	time_5 = get_time();

	//Copy back the extendted image array
	for (int i = 1; i <= heightPerProcess ; i++) {
	  for (int j = 1; j <= width ; j++) {
	    k = i * (width+2) + j;	// position of current element
	    k2 = (i-1) * width + (j-1);
	    image[k2] = (unsigned char)lclimage[k];
	  }
	}
	MPI_Gather(image, heightPerProcess*width, MPI_UNSIGNED_CHAR, tmp_image, heightPerProcess*width, MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == MASTER) {
		
		stbi_write_png(outputname, width, height, pixelWidth, tmp_image, 0);
		time_6 = get_time();
	

		// Part VII: get average of sum of pixels for testing and calculate GFLOPS
		// FOR VALIDATION - DO NOT PARALLELIZE
		double test = 0;
		for (int i = 0; i < height; i++) {
		  for (int j = 0; j < width; j++) {
		    test += tmp_image[i * (width) + j];
		  }
		}
		test /= n_pixels;	


		float gflops = (float) (n_iter * 1E-9 * (3 * height * width + 41 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
		time_7 = get_time();

		// Part VII: deallocate variables
		stbi_image_free(tmp_image);
		free(image);
		free(lclimage);
		free(north_deriv);
		free(south_deriv);		
		free(west_deriv);
		free(east_deriv);
		free(diff_coef);
		free(lcl_diff_coef);
		time_8 = get_time();

		// print
		printf("Time spent in different stages of the application:\n");
		printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
		printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
		printf("%9.6f s => Part III: read image\n", (time_3 - time_2));
		printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
		printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
		printf("%9.6f s => Part VI: write image to file\n", (time_6 - time_5));
		printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
		printf("%9.6f s => Part VIII: deallocate variables\n", (time_7 - time_6));
		printf("Total time: %9.6f s\n", (time_8 - time_0));
		printf("Average of sum of pixels: %9.6f\n", test);
		printf("GFLOPS: %f\n", gflops); 
	}
	free(lclimage);
	free(lcl_diff_coef);
	Finalize();
	return 0;
}

// Update the ghost cells of image at boundary
void update_image_ghostcells(unsigned char *image, int height, int width)
{

  for (int h = 1; h < height-1; h++) {    
    image[h*width + 0] = image[h*width + width-2];
    image[h*width + width-1] = image[h*width + 1];
  }
  
  // for (int w = 1; w < width-1; w++) {
  //   image[0*width + w] = image[(height-2)*width + w];
  //   image[(height-1)*width + w] = image[1*width + w];
  // }
}

// Update the ghost cells of diff_coeff at boundary
void update_coeff_ghostcells(float *diff_coeff, int height, int width)
{

  for (int h = 1; h < height-1; h++) {
    diff_coeff[h*width + width-1] = diff_coeff[h*width + 1];
  }
  
  // for (int w = 1; w < width-1; w++) {
  //   diff_coeff[width*(height-1) + w] = diff_coeff[1*width + w];
  // }
}

void update_image_ghostcells_extended(unsigned char *image, int height, int width)
{

  for (int h = 1; h < height-1; h++) {    
    image[h*width + 0] = image[h*width + width-2];
    image[h*width + width-1] = image[h*width + 1];
  }
  
  for (int w = 1; w < width-1; w++) {
    image[0*width + w] = image[(height-2)*width + w];
    image[(height-1)*width + w] = image[1*width + w];
  }
}

// Update the ghost cells of diff_coeff at boundary
void update_coeff_ghostcells_extended(float *diff_coeff, int height, int width)
{

  for (int h = 1; h < height-1; h++) {
    diff_coeff[h*width + width-1] = diff_coeff[h*width + 1];
  }
  
  for (int w = 1; w < width-1; w++) {
    diff_coeff[width*(height-1) + w] = diff_coeff[1*width + w];
  }
}
