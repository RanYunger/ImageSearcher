// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "Utilities.h"

// Methods
// ======================================= MPI methods =======================================
int countIntegersInImage(Image image)
{
	return 2 + (image.dimension * image.dimension); // ID, dimension + colors in image's color matrix
}
int countIntegersInImagesCollection(Image *imagesArray, int imagesCount)
{
	int integersCount = 1; // for unpacking imagesCount before all the data

	// Calculates the size in integers of the objects array
	for (int i = 0; i < imagesCount; i++)
		integersCount += countIntegersInImage(imagesArray[i]);

	return integersCount + imagesCount; // 1 extra integer per image to store its size in bytes
}

char* packImage(Image image)
{
	int colorsInImage, integersInImage = countIntegersInImage(image),
			sizeInBytes = integersInImage * sizeof(int), packingOffset = 0;
	char *packedBuffer;

	// Allocates memory for the packed image
	packedBuffer = (char*) malloc(integersInImage * sizeof(int));
	if (packedBuffer == NULL)
	{
		printf("\nCannot allocate memory for packed image\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	// Packs the image's ID and dimension
	MPI_Pack(&(image.ID), 1, MPI_INT, packedBuffer, sizeInBytes, &packingOffset, MPI_COMM_WORLD);
	MPI_Pack(&(image.dimension), 1, MPI_INT, packedBuffer, sizeInBytes, &packingOffset, MPI_COMM_WORLD);

	// Packs the image's color matrix
	colorsInImage = image.dimension * image.dimension;
	MPI_Pack(image.colorsMatrix, colorsInImage, MPI_INT, packedBuffer, sizeInBytes, &packingOffset, MPI_COMM_WORLD);

	return packedBuffer;
}
char* packImagesCollection(Image *imagesArray, int imagesCount)
{
	int integersInCollection, integersInImage, sizeInBytes, packingOffset;
	char *packedBuffer, *packedImage;

	// Allocates memory for the packed buffer
	integersInCollection = countIntegersInImagesCollection(imagesArray, imagesCount);
	packedBuffer = (char*) malloc(integersInCollection * sizeof(int));
	if (packedBuffer == NULL)
	{
		printf("\nCannot allocate memory for packed images array\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	sizeInBytes = integersInCollection * sizeof(int);
	packingOffset = 0;

	// Packs imagesCount into the buffer
	MPI_Pack(&imagesCount, 1, MPI_INT, packedBuffer, sizeInBytes, &packingOffset, MPI_COMM_WORLD);

	// Packs the images collection into the buffer
	for (int i = 0; i < imagesCount; i++)
	{
		// Allocates memory for the packed image's buffer
		integersInImage = countIntegersInImage(imagesArray[i]);
		sizeInBytes = integersInImage * sizeof(int);

		// Packs the image's size in bytes
		MPI_Pack(&sizeInBytes, 1, MPI_INT, packedBuffer,
				integersInCollection * sizeof(int), &packingOffset, MPI_COMM_WORLD);

		packedImage = packImage(imagesArray[i]);

		memcpy(packedBuffer + packingOffset, packedImage, sizeInBytes);
		packingOffset += sizeInBytes;
	}

	return packedBuffer;
}
Image unpackImage(char *packedBuffer)
{
	int colorsInImage, sizeInBytes = sizeof(packedBuffer), unpackingOffset = 0;
	Image image;

	// Unpacks the image's ID and dimension
	MPI_Unpack(packedBuffer, sizeInBytes, &unpackingOffset, &(image.ID), 1,
	MPI_INT, MPI_COMM_WORLD);
	MPI_Unpack(packedBuffer, sizeInBytes, &unpackingOffset, &(image.dimension),
			1, MPI_INT,
			MPI_COMM_WORLD);

	// Allocates memory for the unpacked image's color matrix
	colorsInImage = image.dimension * image.dimension;
	image.colorsMatrix = (int*) malloc(colorsInImage * sizeof(int));
	if (image.colorsMatrix == NULL)
	{
		printf("\nCannot allocate memory for unpacked image's color matrix\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	// Unpacks the image's color matrix
	MPI_Unpack(packedBuffer, sizeInBytes, &unpackingOffset, image.colorsMatrix, colorsInImage, MPI_INT, MPI_COMM_WORLD);

	return image;
}
Image* unpackImagesCollection(char *packedBuffer, int *imagesCount)
{
	int sizeInBytes, unpackingOffset;
	char *packedImage;
	Image *imagesArray;

	unpackingOffset = 0;

	// Unpacks amount of images in the collection
	MPI_Unpack(packedBuffer, sizeof(packedBuffer), &unpackingOffset, imagesCount, 1, MPI_INT, MPI_COMM_WORLD);

	// Allocates memory for the unpacked images collection
	imagesArray = (Image*) malloc(*imagesCount * sizeof(Image));
	if (imagesArray == NULL)
	{
		printf("\nCannot allocate memory for unpacked images array\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	// Unpacks the images collection
	for (int i = 0; i < *imagesCount; i++)
	{
		// Unpacks the image's size in bytes
		MPI_Unpack(packedBuffer, sizeof(packedBuffer), &unpackingOffset, &sizeInBytes, 1, MPI_INT, MPI_COMM_WORLD);

		// Allocates memory for the packed image's buffer
		packedImage = (char*) malloc(sizeInBytes * sizeof(char));
		if (packedImage == NULL)
		{
			printf("\nCannot allocate memory for unpacked image's buffer\n");
			MPI_Abort(MPI_COMM_WORLD, 0);
			exit(0);
		}

		memcpy(packedImage, packedBuffer + unpackingOffset, sizeInBytes);
		imagesArray[i] = unpackImage(packedImage);
		unpackingOffset += sizeInBytes;
	}

	return imagesArray;
}

void sendPictureToProcess(Image picture, int receiverProcessID)
{
	int integersInPicture;
	char *packedPicture;

	// Sends the picture's integers count to the receiver process (for memory allocation)
	integersInPicture = countIntegersInImage(picture);
	MPI_Send(&integersInPicture, 1, MPI_INT, receiverProcessID, 0, MPI_COMM_WORLD);

	// Packs the picture and sends it to the receiver process
	packedPicture = packImage(picture);
	MPI_Send(packedPicture, integersInPicture * sizeof(int), MPI_CHAR, receiverProcessID, WORK, MPI_COMM_WORLD);
}
Image recievePictureFromProcess(int senderProcessID)
{
	Image emptyImage;
	int integersInPicture;
	char *packedPicture;
	MPI_Status status;

	// Sets up a default return value (in case TERMINATETAG is met)
	emptyImage.ID = NOTFOUND;
	emptyImage.dimension = NOTFOUND;
	emptyImage.colorsMatrix = NULL;

	// Receives the picture's integers count from the sender process
	MPI_Recv(&integersInPicture, 1, MPI_INT, senderProcessID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	if (status.MPI_TAG == TERMINATE)
		return emptyImage;

	packedPicture = (char*) malloc(integersInPicture * sizeof(int));
	if (packedPicture == NULL)
	{
		printf("\nCannot allocate memory to receive packed image\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	// Receives the packed picture from the sender process
	MPI_Recv(packedPicture, integersInPicture * sizeof(int), MPI_CHAR, senderProcessID, WORK, MPI_COMM_WORLD, &status);

	return unpackImage(packedPicture);
}

// ======================================= OpenMP methods =======================================
SearchRecord searchObjectInPicture(Image picture, int *devicePictureColorsMatrix, Image object, float matchingEpsilon)
{
	int colorsInPicture = picture.dimension * picture.dimension; 
	int *positionFlagsArray;
	SearchRecord result;
		
	// Initializes a default result
	result.pictureID = picture.ID;
	result.objectID = object.ID;
	result.objectPosition.row = NOTFOUND;
	result.objectPosition.column = NOTFOUND;
	
	// Validation - Object's dimension isn't blocked by the picture's
	if (object.dimension > picture.dimension)
		return result;
	
	// Searches the object within the picture using the GPU
	positionFlagsArray = searchOnGPU(picture.dimension, devicePictureColorsMatrix, object, matchingEpsilon);	
	
	// Extracts the final search record - the first position the object was found in (if at all) or "not found"
	for (int i = 0; i < colorsInPicture; i++)
		if (positionFlagsArray[i])
		{
			result.objectPosition.row = i / picture.dimension;
			result.objectPosition.column = i % picture.dimension;
			break;	
		}
		
	return result;
}
SearchRecord searchObjectsInPicture(Image picture, Image *objectsArray, int objectsCount, float matchingEpsilon)
{
	SearchRecord result, searchRecords[objectsCount];
	int *devicePicture = NULL;
	
	// Sets the amount of OpenMP threads to be used
	omp_set_num_threads(objectsCount);

	// Allocates the picture on the GPU (for further calculations)
	allocateImageOnGPU(picture, &devicePicture);
	
	// Searches the objects in the picture using threads - each thread handles an object
	#pragma omp parallel for shared(picture, searchRecords) num_threads(objectsCount)
	for (int threadID = 0; threadID < objectsCount; threadID++)
		searchRecords[threadID] = searchObjectInPicture(picture, devicePicture, objectsArray[threadID], matchingEpsilon);
	
	// Frees the picture from the GPU
	freeImageFromGPU(&devicePicture);	
		
	// Extracts the final result from all search attempts - the first found object (if there is one) or "not found"
	result.pictureID = picture.ID;
	result.objectID = NOTFOUND;
	result.objectPosition.row = NOTFOUND;
	result.objectPosition.column = NOTFOUND;
	
	for (int i = 0; i < objectsCount; i++)
		if (isNotEmptyPosition(searchRecords[i].objectPosition))
		{
			result.objectID = searchRecords[i].objectID;
			result.objectPosition.row = searchRecords[i].objectPosition.row;
			result.objectPosition.column = searchRecords[i].objectPosition.column;
			break;
		}

	return result;
}

// ======================================= Input methods =======================================
void readInputFile(float *matchingEpsilon, int *picturesCount,
		Image **picturesArray, int *objectsCount, Image **objectsArray)
{
	FILE *filePointer;

	// Opens file for reading parameters
	if ((filePointer = fopen(INPUTFILE, "r")) == 0)
	{
		printf("\nCannot open %s for reading\n", INPUTFILE);
		exit(0);
	}

	// Reads matchingEpsilon from file
	fscanf(filePointer, "%f", matchingEpsilon);

	// Reads pictures and objects collections from file
	*picturesArray = readImagesCollectionFromFile(filePointer, picturesCount);
	*objectsArray = readImagesCollectionFromFile(filePointer, objectsCount);

	// Closes file
	fclose(filePointer);

	printf("Reading data from file %s complete\n", INPUTFILE);
}
Image* readImagesCollectionFromFile(FILE *filePointer, int *imagesCount)
{
	Image *imagesArray;

	// Reads images count from file
	fscanf(filePointer, "%d", imagesCount);
	if ((*imagesCount <= 0) || (*imagesCount > LIMIT))
	{
		printf("Input file must store 1-%d images\n", LIMIT);
		exit(0);
	}

	// Allocates memory for images collection
	imagesArray = (Image*) malloc(*imagesCount * sizeof(Image));
	if (imagesArray == NULL)
	{
		printf("Cannot allocate memory for images collection\n");
		exit(0);
	}

	// Reads the image collection from file
	for (int i = 0; i < *imagesCount; i++)
		readImageFromFile(filePointer, imagesArray, i, &(imagesArray[i]));

	return imagesArray;
}
void readImageFromFile(FILE *filePointer, Image *imagesArray,
		int currentImagesCount, Image *image)
{
	int colorsCount;

	// Reads image's ID and dimension from file
	fscanf(filePointer, "%d %d", &image->ID, &image->dimension);
	if (image->ID < 0)
	{
		printf("\nAn image's ID must be positive\n");
		exit(0);
	}
	if (isExistingImage(imagesArray, currentImagesCount, image->ID))
	{
		printf("\nAn image with ID %d already exists\n", image->ID);
		exit(0);
	}

	colorsCount = image->dimension * image->dimension; // N * N colors in image's color matrix

	// Allocates memory for image's color matrix
	image->colorsMatrix = (int*) malloc(colorsCount * sizeof(int));
	if (image->colorsMatrix == NULL)
	{
		printf("cannot allocate memory for image's color matrix\n");
		exit(0);
	}

	// Reads image's color matrix from file
	for (int i = 0; i < colorsCount; i++)
	{
		fscanf(filePointer, "%d", &(image->colorsMatrix[i]));
		if ((image->colorsMatrix[i] < 1) || (image->colorsMatrix[i] > 100))
		{
			printf("A color's range must be [1, 100]\n");
			exit(0);
		}
	}
}

// ======================================= Service methods =======================================
void printColorsMatrix(int *colorsMatrix, int dimension)
{
	int colorsCount = dimension * dimension;

	// Pretty prints the image's color matrix
	for (int i = 0; i < colorsCount; i++)
	{
		if (i % dimension == 0)
			printf("\n\t");
		printf("%d ", colorsMatrix[i]);
	}
	printf("\n");
}
void printImage(Image image)
{
	// Pretty prints the image
	printf("\t[ID: %d | Dimension: %d]\n\tImage:", image.ID, image.dimension);
	printColorsMatrix(image.colorsMatrix, image.dimension);
}
void printAllData(float matchingEpsilon, int picturesCount,
		Image *picturesArray, int objectsCount, Image *objectsArray)
{
	// Prints the matching epsilon
	printf("%s contains the following data:\nmatchingEpsilon = %f\n",
	INPUTFILE, matchingEpsilon);

	// Prints the array of pictures
	printf("Pictures [%d items(s)]:\n", picturesCount);
	for (int i = 0; i < picturesCount; i++)
		printImage(picturesArray[i]);

	// Prints the array of objects
	printf("Objects [%d item(s)]:\n", objectsCount);
	for (int i = 0; i < objectsCount; i++)
		printImage(objectsArray[i]);
}

int isExistingImage(Image *imagesArray, int imagesCount, int imageID)
{
	// Searches an image by the given ID
	for (int i = 0; i < imagesCount; i++)
		if (imagesArray[i].ID == imageID)
			return 1;

	return 0;
}
int isValidPosition(int pictureDimension, int objectDimension, Position position)
{
	// Validations
	// Object's dimension isn't blocked by the picture's
	if (objectDimension > pictureDimension)
		return 0;

	// Object's position exceeds the picture's dimensions
	if (((pictureDimension - position.row) < objectDimension)
			|| ((pictureDimension - position.column) < objectDimension))
		return 0;
		
	return 1;
}
int isNotEmptyPosition(Position position)
{
	return (position.row != NOTFOUND) && (position.column != NOTFOUND);
}

// ======================================= Output methods =======================================
void writeOutputFile(SearchRecord *searchRecords, int recordsCount)
{
	FILE *filePointer;
	int isObjectFound;

	// Open file for reading parameters
	if ((filePointer = fopen(OUTPUTFILE, "w")) == 0)
	{
		printf("\nCannot open %s for writing\n", OUTPUTFILE);
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	// Writes all search records to file
	for (int i = 0; i < recordsCount; i++)
	{
		isObjectFound = searchRecords[i].objectID != NOTFOUND;

		fprintf(filePointer, "Picture #%d: %s", searchRecords[i].pictureID,
				isObjectFound ? "found" : "no objects were found\n");
		if (isObjectFound)
			fprintf(filePointer, " object #%d at (%d, %d)\n",
					searchRecords[i].objectID,
					searchRecords[i].objectPosition.row,
					searchRecords[i].objectPosition.column);
	}

	fclose(filePointer);

	printf("Writing data to file %s complete\n", OUTPUTFILE);
}
