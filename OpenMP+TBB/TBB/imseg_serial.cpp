#include <stdio.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <algorithm>
#include <sys/time.h>

// #include <unordered_map>
#include "tbb/tbb.h"
#include "tbb/atomic.h"

using namespace tbb;

#define MATCH(s) (!strcmp(argv[ac], (s)))




static const double kMicro = 1.0e-6;
double getTime()
{
	struct timeval TV;
	struct timezone TZ;

	const int RC = gettimeofday(&TV, &TZ);
	if(RC == -1) {
		printf("ERROR: Bad call to gettimeofday\n");
		return(-1);
	}

	return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}


class loopNestOne {
	int *m_labels; unsigned char *m_data; int m_width; int m_height; int m_pixelWidth; int m_Threshold;
public:
	void operator()( const blocked_range2d<int>& r) const {
		int *labels = m_labels;
		unsigned char *data = m_data;
		int width = m_width;
		int height = m_height; 
		int pixelWidth = m_pixelWidth; 
		int Threshold = m_Threshold;

		for (int i = r.rows().begin() ; i != r.rows().end() ; ++i) {
			for (int j = r.cols().begin() ; j != r.cols().end() ; ++j) {
				int idx = i*width + j;
				int idx3 = idx*pixelWidth;

				if (labels[idx] == 0)
				  continue;
				  
				int ll = labels[idx]; // save previous label

				// pixels are stored as 3 ints in "data" array. we just use the first of them.
				// Compare with each neighbor:east, west, south, north, ne, nw, se, sw

				//west
				if (j != 0 && abs((int)data[(i*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[i*width + j - 1]);

				//east
				if (j != width-1 && abs((int)data[(i*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[i*width + j + 1]);
				
				//south 
				if(i != height-1 && abs((int)data[((i+1)*width + j)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i+1)*width + j]);

				//north 
				if(i != 0 && abs((int)data[((i-1)*width + j)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i-1)*width + j]);

				//south east
				if(i != height-1 && j != width-1 && abs((int)data[((i+1)*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i+1) * width + j + 1]);

				//north east
				if(i != 0 && j != width-1 && abs((int)data[((i-1)*width + j + 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i-1) * width + j + 1]);

				//south west 
				if(i != height-1 && j!= 0 && abs((int)data[((i+1)*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i+1) * width + j - 1]);

				//north west
				if(i != 0 && j != 0 && abs((int)data[((i-1)*width + j - 1)*pixelWidth] - (int)data[idx3]) < Threshold)
				  labels[idx] = std::max(labels[idx], labels[(i-1) * width + j - 1]);

				// if label assigned to this pixel during this "follow the pointers" step is worse than one of its neighbors, 
				// then that means that we're converging to local maximum instead
				// of global one. To correct this, we replace our root pixel's label with better newly found one.
				if (ll < labels[idx]) {
				  labels[ll - 1] = std::max(labels[idx],labels[ll - 1]);		  
				}
			}
		}
	}

	loopNestOne(int *labels, unsigned char *data, int width, int height, int pixelWidth, int Threshold) : 
		m_labels(labels), m_data(data), m_width(width), m_height(height), m_pixelWidth(pixelWidth), m_Threshold(Threshold) {}
};


void imageSegmentation(int *labels, unsigned char *data, int width, int height, int pixelWidth, int Threshold, int numThreads)
{
	int maxN = std::max(width,height);
	int phases = (int) ceil(log(maxN)/log(2)) + 1;
	int converge = 1;
	
  	for(int pp = 0; pp <= phases; pp++)
	  {
	    converge = 1;
		//LOOP NEST 1
	    // first pass over the image: Find neighbors with better labels.
	    int iter = height/numThreads;
	  	parallel_for(blocked_range2d<int>(0, height, iter, 0, width, width), loopNestOne(labels, data, width, height, pixelWidth, Threshold));
	    
	    //LOOP NEST 2
	    // Second pass on the labels. propagates the updated label of the parent to the children.
	    for (int i = 0; i < height; i++) {
	      for (int j = 0; j < width; j++) {

		int idx = i*width + j;

		if (labels[idx] != 0) {
		  int ll = labels[idx];
		  labels[idx] = std::max(labels[idx], labels[labels[idx] - 1]);
		  if(ll != labels[idx])
			  	converge = 0;
		  // subtract 1 from pixel's label to convert it to array index
		}
	      }
	    }
	    if(converge==1)
	    	break;
	  }
}

int main(int argc,char **argv)
{
	int width,height;
	int pixelWidth;
	int Threshold = 3;
	int numThreads = 1;
	int seed =1 ;
	const char *filename = "input.png";
	const char *outputname = "output.png";
	task_scheduler_init tbb_init;

	// Parse command line arguments
	if(argc<2)
	  {
	    printf("Usage: %s [-i < filename>] [-s <threshold>] [-t <numThreads>] [-o outputfilename]\n",argv[0]);
	    return(-1);
	  }
	for(int ac=1;ac<argc;ac++)
	  {
	    if(MATCH("-s")) {Threshold = atoi(argv[++ac]);}
	    else if(MATCH("-t")) {numThreads = atoi(argv[++ac]);}
	    else if(MATCH("-i"))  {filename = argv[++ac];}
	    else if(MATCH("-o"))  {outputname = argv[++ac];}
	    else {
	      printf("Usage: %s [-i < filename>] [-s <threshold>] [-t <numThreads>] [-o outputfilename]\n",argv[0]);
	      return(-1);
	    }
	  }
	
	printf("Reading image...\n");
	unsigned char *data = stbi_load(filename, &width, &height, &pixelWidth, 0);
	if (!data) {
		fprintf(stderr, "Couldn't load image.\n");
		return (-1);
	}

	printf("Image Read. Width : %d, Height : %d, nComp: %d\n",width,height,pixelWidth);

	int *labels = (int *)malloc(sizeof(int)*width*height);
	unsigned char *seg_data = (unsigned char *)malloc(sizeof(unsigned char)*width*height*3);

	printf("Applying segmentation...\n");

	double start_time = getTime(); 

	//Intially each pixel has a different label
	for(int i = 0; i < height; i++)
	  {
	    for(int j = 0; j < width; j++)
	      {
		int idx = (i*width+j);
		int idx3 = idx*pixelWidth;

		labels[idx] = 0;

		//comment this line if you want to label background pixels as well
		if((int)data[idx3] == 0) 
		  continue;

		//labels are positive integers
		labels[idx] = idx + 1;
	      }
	  }
	
	//Now perform relabeling 
	imageSegmentation(labels,data,width,height,pixelWidth,Threshold,numThreads);
	
	double stop_time = getTime();
	double segTime = stop_time - start_time;	

	int *red = (int *)malloc(sizeof(int)*width*height);
	int *green = (int *)malloc(sizeof(int)*width*height);
	int *blue = (int *)malloc(sizeof(int)*width*height);
	atomic<int> *count = (atomic<int> *)malloc(sizeof(int)*width*height);

	
	parallel_for(0, height * width, [=](int i) {
		count[i] = 0;
		red[i] = -1;
		blue[i] = -1;
		green[i] = -1;	
	});	

	srand(seed);
	start_time = getTime();
	int clusters = 0;
	int min_cluster = height*width;
	int max_cluster = -1;
	double avg_cluster = 0.0;

	parallel_for(0, height*width, [=](int i) {
		if (labels[i] != 0) {
			count[labels[i]-1]++;
		}
	});	

	for (int i = 0 ; i < height*width ; i++) {
		if (labels[i] == 0) {
			
		} else {
			if ((count[i] != 0) && (red[labels[i]-1] == -1)) {
				clusters++;
				red[labels[i]-1] = (int)random()*255;
				green[labels[i]-1] = (int)random()*255;
				blue[labels[i]-1] = (int)random()*255;
			}
		}
	}
	
    // #pragma omp parallel for
    parallel_for(0, height*width, [=](int i) {
		int label = labels[i];
	    seg_data[(i)*3+0] = (char)red[label-1];
	    seg_data[(i)*3+1] = (char)blue[label-1];
	    seg_data[(i)*3+2] = (char)green[label-1];	        	
    });	

	for (int i = 0 ; i < height*width ; i++) {
	  	if (count[i] != 0) {
		    min_cluster = std::min( min_cluster, (int)count[i]);
		    max_cluster = std::max( max_cluster, (int)count[i]);
		    avg_cluster += count[i];
		}
	}

	
	stop_time = getTime();
	double colorTime = stop_time - start_time;
	
	printf("Segmentation Time (sec): %f\n", segTime);
	printf("Coloring Time     (sec): %f\n", colorTime);
	printf("Total Time        (sec): %f\n", colorTime + segTime);
	printf("-----------Statisctics---------------\n");
	printf("Number of Clusters   : %d\n", clusters);
	printf("Min Cluster Size     : %d\n", min_cluster);
	printf("Max Cluster Size     : %d\n", max_cluster);
	printf("Average Cluster Size : %f\n", avg_cluster/clusters);
		
	printf("Writing Segmented Image...\n");
	stbi_write_png(outputname, width, height, 3, seg_data, 0);
	stbi_image_free(data);
	free(seg_data);
	free(labels);

	printf("Done...\n");
	return 0;
}
