#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "math.h"
#include "Utilities.h"

// ======================================= Service methods =======================================
__host__ void allocateImageOnGPU(Image image, int **deviceImage) 
{
	int colorsInImage = image.dimension * image.dimension;
	
	cudaError_t error = cudaSuccess;
		
	// Allocates and copies the object to GPU
	error = cudaMalloc(deviceImage, colorsInImage * sizeof(int));
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for image: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
  	error = cudaMemcpy(*deviceImage, image.colorsMatrix, colorsInImage * sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		printf("Cannot copy image to GPU: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
}
__host__ void freeImageFromGPU(int **deviceImage) 
{
	cudaError_t error = cudaSuccess;
	
	// Frees the picture from GPU memory
	error = cudaFree(*deviceImage);
	if (error != cudaSuccess)
	{
		printf("Cannot free image from GPU: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
}

// ======================================= Device methods =======================================
__device__ __host__ int getPositionsPerDimension(int pictureDimension, int objectDimension) // called from both CPU and GPU
{
	return (pictureDimension - objectDimension) + 1;
}

__device__ float difference(int p, int o) 
{
	return abs(((float)p - o) / p);
}

// ======================================= Kernel methods =======================================
__global__ void searchPositions(int pictureDimension, int *devicePictureColorsMatrix,
	int objectDimension, int *deviceObjectColorsMatrix, float *deviceMatchingsArray,
	int *devicePositionFlagsArray, float matchingEpsilon) 
{
	int threadsPerDimension = getPositionsPerDimension(pictureDimension, objectDimension) * objectDimension;
	int isWithinPicture = (threadIdx.x < threadsPerDimension) && (threadIdx.y < threadsPerDimension);
	int threadID, pictureOffset, objectOffset, matchingOffset;
	
	// Checks threads position in relation to the required amount of threads
	if (isWithinPicture)
	{
		// Initializes required variables
		threadID = (threadIdx.y * threadsPerDimension) + threadIdx.x;
		pictureOffset = threadID % (pictureDimension * pictureDimension);
		objectOffset = threadID % (objectDimension * objectDimension);
		matchingOffset = threadID / (objectDimension * objectDimension);
	
		// Calculates difference value and adds it to the right matching
		atomicAdd(&(deviceMatchingsArray[matchingOffset]), difference(devicePictureColorsMatrix[pictureOffset], deviceObjectColorsMatrix[objectOffset]));
		__syncthreads();
		
		// Converts calculated matching values into flags
		if (objectOffset == 0)
			devicePositionFlagsArray[matchingOffset] = (deviceMatchingsArray[matchingOffset] != 0) && (deviceMatchingsArray[matchingOffset] < matchingEpsilon);
		__syncthreads();		
	}
}

// ======================================= Entry Point =======================================
__host__ int* searchOnGPU(int pictureDimension, int *devicePictureColorsMatrix, Image object, float matchingEpsilon) 
{
	int positionsPerDimension = getPositionsPerDimension(pictureDimension, object.dimension), positionsCount = pow(positionsPerDimension, 2);
	int differencesPerDimension = positionsPerDimension * object.dimension;
	int blocksPerDimension = (differencesPerDimension / BLOCKDIMENSION) + ((differencesPerDimension / BLOCKDIMENSION) != 0);  
	int *hostPositionFlagsArray, *devicePositionFlagsArray, *deviceObjectColorsMatrix;
	float *deviceMatchingsArray;

	dim3 gridDimensions(blocksPerDimension, blocksPerDimension);
	dim3 blockDimensions(BLOCKDIMENSION, BLOCKDIMENSION);
	cudaError_t error = cudaSuccess;
		
	// Allocates memory for the position flags array
	hostPositionFlagsArray = (int*)malloc(positionsCount * sizeof(int));
	if (hostPositionFlagsArray == NULL) 
	{
		printf("Cannot allocate meory for position flags array\n");
		exit(0);
	}
	
	// Allocates and initializes required variables on the GPU
	allocateImageOnGPU(object, &deviceObjectColorsMatrix);
	
	error = cudaMalloc(&devicePositionFlagsArray, positionsCount * sizeof(int));
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for position flags array: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
  	
  	error = cudaMalloc(&deviceMatchingsArray, positionsCount * sizeof(float));
	if (error != cudaSuccess)
	{
		printf("Cannot allocate GPU memory for matchings array: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
  	
  	error = cudaMemset(deviceMatchingsArray, 0, positionsCount * sizeof(float));
	if (error != cudaSuccess)
	{
		printf("Cannot initialize matchings array on GPU: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
  	
  	// Searches the object in the picture using CUDA - each block searches 1024 positions in the picture
  	searchPositions<<<gridDimensions, blockDimensions>>>(pictureDimension, devicePictureColorsMatrix,
  		object.dimension, deviceObjectColorsMatrix, deviceMatchingsArray, devicePositionFlagsArray, matchingEpsilon);
	
	// Copies the position flags array from GPU to host
	error = cudaMemcpy(hostPositionFlagsArray, devicePositionFlagsArray, positionsCount * sizeof(int), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		printf("Cannot copy position flags from GPU to host: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
	
	// Frees allocated variables from the GPU
	error = cudaFree(deviceMatchingsArray);
	if (error != cudaSuccess)
	{
		printf("Cannot free matchings array from GPU: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
	
	error = cudaFree(devicePositionFlagsArray);
	if (error != cudaSuccess)
	{
		printf("Cannot free position flags array from GPU: %s (%d)\n", cudaGetErrorString(error), error);
    		exit(0);
  	}
	
	freeImageFromGPU(&deviceObjectColorsMatrix);
	
	return hostPositionFlagsArray;
}
