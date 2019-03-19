/*	
 * noise_remover.cpp
 *
 * This program removes noise from an image based on Speckle Reducing Anisotropic Diffusion
 * Y. Yu, S. Acton, Speckle reducing anisotropic diffusion, 
 * IEEE Transactions on Image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
 * Original implementation is Modified by Burak BASTEM
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

int main(int argc, char *argv[]) {
	// Part I: allocate and initialize variables
	double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
	time_0 = get_time();
	const char *filename = "input.pgm";
	const char *outputname = "output.png";	
	int width, height, pixelWidth, n_pixels;
	int n_iter = 50;
	double lambda = 0.5;
	double mean, variance, std_dev;	//local region statistics
	double *north_deriv, *south_deriv, *west_deriv, *east_deriv;	// directional derivatives
	double tmp, sum, sum2;	// calculation variables
	double gradient_square, laplacian, num, den, std_dev2, divergence;	// calculation variables
	double *diff_coef;	// diffusion coefficient
	double diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east;	// directional diffusion coefficients
	long k;	// current pixel index
	time_1 = get_time();	
	
	// Part II: parse command line arguments
	if(argc<2) {
	  printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
	  return(-1);
	}
	for(int ac=1;ac<argc;ac++) {
		if(MATCH("-i")) {
			filename = argv[++ac];
		} else if(MATCH("-iter")) {
			n_iter = atoi(argv[++ac]);
		} else if(MATCH("-l")) {
			lambda = atof(argv[++ac]);
		} else if(MATCH("-o")) {
			outputname = argv[++ac];
		} else {
		printf("Usage: %s [-i < filename>] [-iter <n_iter>] [-l <lambda>] [-o <outputfilename>]\n",argv[0]);
		return(-1);
		}
	}
	time_2 = get_time();

	// Part III: read image	
	printf("Reading image...\n");
	unsigned char *image = stbi_load(filename, &width, &height, &pixelWidth, 0);
	if (!image) {
		fprintf(stderr, "Couldn't load image.\n");
		return (-1);
	}
	printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
	n_pixels = height * width;
	time_3 = get_time();

	// Part IV: allocate variables
	north_deriv = (double*) malloc(sizeof(double) * n_pixels);	// north derivative
	south_deriv = (double*) malloc(sizeof(double) * n_pixels);	// south derivative
	west_deriv = (double*) malloc(sizeof(double) * n_pixels);	// west derivative
	east_deriv = (double*) malloc(sizeof(double) * n_pixels);	// east derivative
	diff_coef  = (double*) malloc(sizeof(double) * n_pixels);	// diffusion coefficient
	time_4 = get_time();

	// Part V: compute --- n_iter * (3 * height * width + 42 * (height-1) * (width-1) + 6) doubleing point arithmetic operations in totaL
	for (int iter = 0; iter < n_iter; iter++) {
		sum = 0;
		sum2 = 0;
		// REDUCTION AND STATISTICS
		// --- 3 doubleing point arithmetic operations per element -> 3*height*width in total
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				tmp = image[i * width + j];	// current pixel value
				sum += tmp; // --- 1 doubleing point arithmetic operations
				sum2 += tmp * tmp; // --- 2 doubleing point arithmetic operations
			}
		}
		printf("Sum: %f, Sum2: %f\n", sum, sum2);
		mean = sum / n_pixels; // --- 1 doubleing point arithmetic operations
		variance = (sum2 / n_pixels) - mean * mean; // --- 3 doubleing point arithmetic operations
		std_dev = variance / (mean * mean); // --- 2 doubleing point arithmetic operations

		//COMPUTE 1
		// --- 32 doubleing point arithmetic operations per element -> 32*(height-1)*(width-1) in total
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				k = i * width + j;	// position of current element
				north_deriv[k] = image[(i - 1) * width + j] - image[k];	// north derivative --- 1 doubleing point arithmetic operations
				south_deriv[k] = image[(i + 1) * width + j] - image[k];	// south derivative --- 1 doubleing point arithmetic operations
				west_deriv[k] = image[i * width + (j - 1)] - image[k];	// west derivative --- 1 doubleing point arithmetic operations
				east_deriv[k] = image[i * width + (j + 1)] - image[k];	// east derivative --- 1 doubleing point arithmetic operations
				gradient_square = (north_deriv[k] * north_deriv[k] + south_deriv[k] * south_deriv[k] + west_deriv[k] * west_deriv[k] + east_deriv[k] * east_deriv[k]) / (image[k] * image[k]); // 9 doubleing point arithmetic operations
				laplacian = (north_deriv[k] + south_deriv[k] + west_deriv[k] + east_deriv[k]) / image[k]; // 4 doubleing point arithmetic operations
				num = (0.5f * gradient_square) - ((1.0f / 16.0f) * (laplacian * laplacian)); // 5 doubleing point arithmetic operations
				den = 1 + (.25f * laplacian); // 2 doubleing point arithmetic operations
				std_dev2 = num / (den * den); // 2 doubleing point arithmetic operations
				den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 doubleing point arithmetic operations
				diff_coef[k] = 1.0f / (1.0f + den); // 2 doubleing point arithmetic operations
				if (diff_coef[k] < 0) {
					diff_coef[k] = 0;
				} else if (diff_coef[k] > 1)	{
					diff_coef[k] = 1;
				}
			}
		}
		// COMPUTE 2
		// divergence and image update --- 10 doubleing point arithmetic operations per element -> 10*(height-1)*(width-1) in total
		for (int i = 1; i < height - 1; i++) {
			for (int j = 1; j < width - 1; j++) {
				k = i * width + j;	// get position of current element
				diff_coef_north = diff_coef[k];	// north diffusion coefficient
				diff_coef_south = diff_coef[(i + 1) * width + j];	// south diffusion coefficient
				diff_coef_west = diff_coef[k];	// west diffusion coefficient
				diff_coef_east = diff_coef[i * width + (j + 1)];	// east diffusion coefficient				
				divergence = diff_coef_north * north_deriv[k] + diff_coef_south * south_deriv[k] + diff_coef_west * west_deriv[k] + diff_coef_east * east_deriv[k]; // --- 7 doubleing point arithmetic operations
				image[k] = image[k] + 0.25f * lambda * divergence; // --- 3 doubleing point arithmetic operations
			}
		}
	}
	time_5 = get_time();

	// Part VI: write image to file
	stbi_write_png(outputname, width, height, pixelWidth, image, 0);
	time_6 = get_time();

	// Part VII: get average of sum of pixels for testing and calculate GFLOPS
	// FOR VALIDATION - DO NOT PARALLELIZE
	double test = 0.0;
	for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				test += image[i * width + j];
				// test += 1.0;
		}
	}
	test /= n_pixels;	
	// printf("h: %d\n", height);
	// printf("w: %d\n", width);
	// printf("w*h: %d\n", width*height);
	// printf("test: %f\n", test);

	double gflops = (double) (n_iter * 1E-9 * (3 * height * width + 42 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
	time_7 = get_time();

	// Part VII: deallocate variables
	stbi_image_free(image);
	free(north_deriv);
	free(south_deriv);
	free(west_deriv);
	free(east_deriv);
	free(diff_coef);
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
	return 0;
}

