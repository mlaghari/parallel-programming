/*	
 * noise_remover.cpp
 *
 * This program removes noise from an h_image based on Speckle Reducing Anisotropic Diffusion
 * Y. Yu, S. Acton, Speckle reducing anisotropic diffusion, 
 * IEEE Transactions on h_image Processing 11(11)(2002) 1260-1270 <http://people.virginia.edu/~sc5nf/01097762.pdf>
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

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
const int threadsPerBlock = 512;
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

__global__ void
reductionAndStatistics(unsigned char *d_image, double *sum, double *sum2, int imageSize) {
	__shared__ double d_sum[threadsPerBlock];
	__shared__ double d_sum2[threadsPerBlock];
	long blockSize = blockDim.x * blockDim.y;
	long offset = threadIdx.x + blockIdx.x * threadsPerBlock;
	int cacheIndex = threadIdx.x;
	if (offset < imageSize) {
		double tmp = d_image[offset];
		d_sum[cacheIndex] = tmp; // --- 1 doubleing point arithmetic operations
		d_sum2[cacheIndex] = tmp*tmp; // --- 2 doubleing point arithmetic operations
		__syncthreads();
		if (blockSize >= 512) {
			if (cacheIndex < 256) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 256];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 256];				
			}
			__syncthreads();
		}

		if (blockSize >= 256) {
			if (cacheIndex < 128) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 128];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 128];				
			}
			__syncthreads();
		}

		if (blockSize >= 128) {
			if (cacheIndex < 64) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 64];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 64];				
			}
			__syncthreads();
		}

		if (cacheIndex < 32) {	
			if (blockSize >= 64)  {
				d_sum[cacheIndex] += d_sum[cacheIndex + 32];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 32]; 
			}
			if (blockSize >= 32)  {
				d_sum[cacheIndex] += d_sum[cacheIndex + 16];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 16]; 
			}
			if (blockSize >= 16) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 8];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 8]; 
			}
			if (blockSize >= 8) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 4];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 4]; 
			}
			if (blockSize >= 4) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 2];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 2]; 
			}
			if (blockSize >= 2) {
				d_sum[cacheIndex] += d_sum[cacheIndex + 1];
				d_sum2[cacheIndex] += d_sum2[cacheIndex + 1]; 
			}
		}
	}

	if (cacheIndex == 0) {
		sum[blockIdx.x] = d_sum[0];
		sum2[blockIdx.x] = d_sum2[0];
	}
}

__global__ void
Compute2(unsigned char *d_image, double *d_north_deriv, double *d_south_deriv, double *d_east_deriv, double *d_west_deriv, double *d_diff_coef, double d_lambda, int width, int height) {
	long threadsPerRow = width;
	long totalThreads = height*width;
	long k = threadIdx.x + blockIdx.x * threadsPerBlock;	// get position of current element
	double diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east, divergence;	// directional diffusion coefficients
	double d_diff_coef_register;
    
    if (k >= threadsPerRow && k < totalThreads-threadsPerRow) {
    	if (((k % threadsPerRow) != 0) && ((k+1) % threadsPerRow) != 0) {
    		d_diff_coef_register = d_diff_coef[k];
			diff_coef_north = d_diff_coef_register;						// north diffusion coefficient
		    diff_coef_south = d_diff_coef[k+threadsPerRow];			// south diffusion coefficient
	    	diff_coef_west = d_diff_coef_register;						// west diffusion coefficient
	    	diff_coef_east = d_diff_coef[k + 1];					// east diffusion coefficient

	    	divergence = diff_coef_north * d_north_deriv[k] + diff_coef_south * 
	    				d_south_deriv[k] + diff_coef_west * d_west_deriv[k] + diff_coef_east * d_east_deriv[k]; // --- 7 doubleing point arithmetic operations
	    	d_image[k] = d_image[k] + 0.25f * d_lambda * divergence; // --- 3 doubleing point arithmetic operations
	    }
    }
}

__global__ void
Compute1(unsigned char *d_image, double std_dev, double *d_north_deriv, double *d_south_deriv, double *d_east_deriv, double *d_west_deriv, double *d_diff_coef, int width, int height, int blocksPerGrid) {
	long k = threadIdx.x + blockIdx.x * threadsPerBlock;
	long totalThreads = height*width;	//17915904
	long threadsPerRow = width;
	double d_image_temp = d_image[k];
	double gradient_square, laplacian, num, den, std_dev2;	// calculation variables
	double north_deriv_value, south_deriv_value, east_deriv_value, west_deriv_value;
	__shared__ unsigned char img_data[3][threadsPerBlock];
	__shared__ unsigned char img_data_before, img_data_after;
	unsigned char img_data_register;	
	
	if (k >= threadsPerRow && k < totalThreads-threadsPerRow) {
		if (threadIdx.x == 0) {
			img_data_before = d_image[(blockIdx.x-1)*threadsPerBlock + (threadsPerBlock - 1)];
		}

		if (threadIdx.x == threadsPerBlock-1) {
			img_data_after = d_image[(blockIdx.x+1)*threadsPerBlock];
		}

		img_data[0][threadIdx.x] = d_image[k - threadsPerRow];
		img_data[1][threadIdx.x] = d_image[k];
		img_data[2][threadIdx.x] = d_image[k + threadsPerRow];
		__syncthreads();

		if (((k % threadsPerRow) != 0) && ((k+1) % threadsPerRow) != 0) {
			img_data_register = img_data[1][threadIdx.x];

			// north derivative --- 1 doubleing point arithmetic operations
			d_north_deriv[k] = img_data[0][threadIdx.x] - img_data_register;	
			north_deriv_value = d_north_deriv[k];

			// south derivative --- 1 doubleing point arithmetic operations
			d_south_deriv[k] = img_data[2][threadIdx.x] - img_data_register;	
			south_deriv_value = d_south_deriv[k];

			// west derivative --- 1 doubleing point arithmetic operations
			if (threadIdx.x != 0) {
				d_west_deriv[k] = img_data[1][threadIdx.x - 1] - img_data_register;	
			} else {
				d_west_deriv[k] = img_data_before - img_data_register;
			}
			west_deriv_value = d_west_deriv[k];

			// east derivative --- 1 doubleing point arithmetic operations
			if (threadIdx.x != threadsPerBlock - 1) {
				d_east_deriv[k] = img_data[1][threadIdx.x + 1] - img_data_register;	
			} else {
				d_east_deriv[k] = img_data_after - img_data_register;
			}
			east_deriv_value = d_east_deriv[k];


			gradient_square = (north_deriv_value * north_deriv_value + south_deriv_value * south_deriv_value + 
								west_deriv_value * west_deriv_value + east_deriv_value * east_deriv_value) 
								/ (d_image_temp * d_image_temp); // 9 doubleing point arithmetic operations

			laplacian = (north_deriv_value + south_deriv_value + west_deriv_value + east_deriv_value) / d_image_temp; // 4 doubleing point arithmetic operations
			
			num = (0.5f * gradient_square) - ((1.0f / 16.0f) * (laplacian * laplacian)); // 5 doubleing point arithmetic operations
			den = 1 + (.25f * laplacian); // 2 doubleing point arithmetic operations
			std_dev2 = num / (den * den); // 2 doubleing point arithmetic operations
			den = (std_dev2 - std_dev) / (std_dev * (1 + std_dev)); // 4 doubleing point arithmetic operations
			d_diff_coef[k] = 1.0f / (1.0f + den); // 2 doubleing point arithmetic operations
			if (d_diff_coef[k] < 0) {
				d_diff_coef[k] = 0;
			} else if (d_diff_coef[k] > 1)	{
				d_diff_coef[k] = 1;
			}
			// __syncthreads();
		}
	}
}


void getGPUInfo() {
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	printf("Installed GPUs: %d\n", count);
	for (int i=0; i< count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf( " --- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
		if (prop.deviceOverlap)
		printf( "Enabled\n" );
		else
		printf( "Disabled\n" );
		printf( "Kernel execition timeout : " );
		if (prop.kernelExecTimeoutEnabled)
		printf( "Enabled\n" );
		else
		printf( "Disabled\n" );
		printf( " --- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( " --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",
		prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n",
		prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1],
		prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d)\n",
		prop.maxGridSize[0], prop.maxGridSize[1],
		prop.maxGridSize[2] );
		printf( "\n" );
	}
}

int main(int argc, char *argv[]) {
	// Part I: allocate and initialize variables
	double time_0, time_1, time_2, time_3, time_4, time_5, time_6, time_7, time_8;	// time variables
	time_0 = get_time();
	const char *filename = "input.pgm";
	const char *outputname = "output.png";	
	
	int n_iter = 50;
	double lambda = 0.5;
	double mean, variance, std_dev;	//local region statistics
	double *north_deriv, *south_deriv, *west_deriv, *east_deriv;	// directional derivatives
	double tmp, sum, sum2;	// calculation variables
	double gradient_square, laplacian, num, den, std_dev2, divergence;	// calculation variables
	double *diff_coef;	// diffusion coefficient
	double diff_coef_north, diff_coef_south, diff_coef_west, diff_coef_east;	// directional diffusion coefficients
	long k;	// current pixel index
	int width = 0;
	int height = 0;
	int pixelWidth = 0;
	int n_pixels = 0;
	int blocksPerGrid = 0;
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

	// CUDA variables
	// Error code: to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // getGPUInfo();

	// Part III: read h_image	
	printf("Reading h_image...\n");
	unsigned char *h_image = stbi_load(filename, &width, &height, &pixelWidth, 0);
	size_t size = height*width;
    blocksPerGrid = (size + (threadsPerBlock - 1))/threadsPerBlock;

	unsigned char *d_image = NULL;
	err = cudaMalloc((void **)&d_image, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (Error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	if (!h_image) {
		fprintf(stderr, "Couldn't load h_image.\n");
		return (-1);
	}
	// printf("h_image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);
	n_pixels = height * width;
	time_3 = get_time();

	// Part IV: allocate variables
	north_deriv = (double*) malloc(sizeof(double) * n_pixels);	// north derivative
	south_deriv = (double*) malloc(sizeof(double) * n_pixels);	// south derivative
	west_deriv = (double*) malloc(sizeof(double) * n_pixels);	// west derivative
	east_deriv = (double*) malloc(sizeof(double) * n_pixels);	// east derivative
	diff_coef  = (double*) malloc(sizeof(double) * n_pixels);	// diffusion coefficient

	double *d_partial_sum, *d_partial_sum2;
	cudaMalloc((void**)&d_partial_sum, blocksPerGrid*sizeof(double));
	cudaMalloc((void**)&d_partial_sum2, blocksPerGrid*sizeof(double));

	double *partial_sum = (double*)malloc(blocksPerGrid*sizeof(double));   
	double *partial_sum2 = (double*)malloc(blocksPerGrid*sizeof(double));

	time_4 = get_time();
	// Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    // printf("Copy h_image from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_image from host to device (Error code:: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Compute 1: Allocating variables for device 
	double *d_north_deriv, *d_south_deriv, *d_west_deriv, *d_east_deriv, *d_diff_coef;
	cudaMalloc((void **)&d_north_deriv, sizeof(double) * n_pixels);	// north derivative
	cudaMalloc((void **)&d_south_deriv, sizeof(double) * n_pixels);	// south derivative
	cudaMalloc((void **)&d_west_deriv, sizeof(double) * n_pixels);	// west derivative
	cudaMalloc((void **)&d_east_deriv, sizeof(double) * n_pixels);	// east derivative
	cudaMalloc((void **)&d_diff_coef, sizeof(double) * n_pixels);	// diffusion coefficient

	// Part V: compute --- n_iter * (3 * height * width + 42 * (height-1) * (width-1) + 6) doubleing point arithmetic operations in totaL
	for (int iter = 0; iter < n_iter; iter++) {
		sum = 0;
		sum2 = 0;

		// Kernel Launch	
		reductionAndStatistics<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_partial_sum, d_partial_sum2, height*width);
		err = cudaGetLastError();
	    if (err != cudaSuccess) {
	        fprintf(stderr, "Failed to launch reductionAndStatistics kernel (Error code: %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
	    cudaMemcpy(partial_sum2, d_partial_sum2, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
	    
	    // cudaDeviceSynchronize();

	    for (int i = 0 ; i < blocksPerGrid ; i++) {
	    	sum += partial_sum[i];
	    	sum2 += partial_sum2[i];
	    }

	    // printf("Sum: %f Sum2: %f\n", sum, sum2);

		mean = sum / n_pixels; // --- 1 doubleing point arithmetic operations
		variance = (sum2 / n_pixels) - mean * mean; // --- 3 doubleing point arithmetic operations
		std_dev = variance / (mean * mean); // --- 2 doubleing point arithmetic operations

		// Compute 1: Kernel Launch 
		Compute1<<<blocksPerGrid, threadsPerBlock>>>(d_image, std_dev, d_north_deriv, d_south_deriv, d_west_deriv, d_east_deriv, d_diff_coef, width, height, blocksPerGrid);
		err = cudaGetLastError();
	    if (err != cudaSuccess) {
	        fprintf(stderr, "Failed to launch Compute1 kernel (Error code: %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    cudaDeviceSynchronize();

	    // Compute 2: Kernel launch
	    Compute2<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_north_deriv, d_south_deriv, d_west_deriv, d_east_deriv, d_diff_coef, lambda, width, height);
		err = cudaGetLastError();
	    if (err != cudaSuccess) {
	        fprintf(stderr, "Failed to launch Compute2 kernel (Error code: %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	    cudaDeviceSynchronize();

	}
	err = cudaMemcpy(h_image, d_image, size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
	    fprintf(stderr, "Failed to copy h_image from host to device (Error code:: %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}
	time_5 = get_time();

	// Part VI: write h_image to file
	stbi_write_png(outputname, width, height, pixelWidth, h_image, 0);
	time_6 = get_time();

	// Part VII: get average of sum of pixels for testing and calculate GFLOPS
	// FOR VALIDATION - DO NOT PARALLELIZE
	double test = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			test += h_image[i * width + j];
		}
	}
	test /= n_pixels;	

	double gflops = (double) (n_iter * 1E-9 * (3 * height * width + 42 * (height-1) * (width-1) + 6)) / (time_5 - time_4);
	time_7 = get_time();

	// Part VII: deallocate variables
	stbi_image_free(h_image);
	free(north_deriv);
	free(south_deriv);
	free(west_deriv);
	free(east_deriv);
	free(diff_coef);
	cudaFree(d_image);
	cudaFree(d_north_deriv);
	cudaFree(d_south_deriv);
	cudaFree(d_west_deriv);
	cudaFree(d_east_deriv);
	cudaFree(d_diff_coef);
	cudaFree(d_partial_sum);
	cudaFree(d_partial_sum2);
	time_8 = get_time();

	// print
	printf("Time spent in different stages of the application:\n");
	printf("%9.6f s => Part I: allocate and initialize variables\n", (time_1 - time_0));
	printf("%9.6f s => Part II: parse command line arguments\n", (time_2 - time_1));
	printf("%9.6f s => Part III: read h_image\n", (time_3 - time_2));
	printf("%9.6f s => Part IV: allocate variables\n", (time_4 - time_3));
	printf("%9.6f s => Part V: compute\n", (time_5 - time_4));
	printf("%9.6f s => Part VI: write h_image to file\n", (time_6 - time_5));
	printf("%9.6f s => Part VII: get average of sum of pixels for testing and calculate GFLOPS\n", (time_7 - time_6));
	printf("%9.6f s => Part VIII: deallocate variables\n", (time_7 - time_6));
	printf("Total time: %9.6f s\n", (time_8 - time_0));
	printf("Average of sum of pixels: %9.6f\n", test);
	printf("GFLOPS: %f\n", gflops);
	return 0;
}

