#pragma once

// Defines
#define INPUTFILE "input.txt"
#define OUTPUTFILE "output.txt"

#define WORK 0
#define TERMINATE 1

#define BLOCKDIMENSION 32					// for CUDA calculations
#define THREADSPERBLOCK BLOCKDIMENSION * BLOCKDIMENSION 	// for CUDA calculations

#define NOTFOUND -1
#define LIMIT 1000

// Structs
struct ImageStruct 
{
	int ID;
	int dimension;
	int *colorsMatrix;
};
typedef struct ImageStruct Image;

struct PositionStruct
{
	int row;
	int column;
};
typedef struct PositionStruct Position;

struct SearchRecordStruct
{
	int pictureID;
	int objectID;						// default value: NOTFOUND
	Position objectPosition;    				// default value: (NOTFOUND, NOTFOUND)
};
typedef struct SearchRecordStruct SearchRecord;

// Methods
// ======================================= External methods =======================================
extern void allocateImageOnGPU(Image image, int **deviceImage);
extern void freeImageFromGPU(int **deviceImage);
extern int* searchOnGPU(int pictureDimension, int *devicePictureColorsMatrix, Image object, float matchingEpsilon);

// ======================================= MPI methods =======================================
int countIntegersInImage(Image image);
int countIntegersInImagesCollection(Image *imagesArray, int imagesCount);

char* packImage(Image image);
char* packImagesCollection(Image *imagesArray, int imagesCount);
Image unpackImage(char *packedBuffer);
Image* unpackImagesCollection(char *packedBuffer, int *imagesCount);

void sendPictureToProcess(Image picture, int receiverProcessID);
Image recievePictureFromProcess(int senderProcessID);

// ======================================= OpenMP methods =======================================
SearchRecord searchObjectInPicture(Image picture, int *devicePictureColorsMatrix, Image object, float matchingEpsilon);
SearchRecord searchObjectsInPicture(Image picture, Image *objectsArray, int objectsCount, float matchingEpsilon);
		
// ======================================= Input methods =======================================
void readInputFile(float *matchingEpsilon, int *picturesCount,
		Image **picturesArray, int *objectsCount, Image **objectsArray);
Image* readImagesCollectionFromFile(FILE *filePointer, int *imagesCount);
void readImageFromFile(FILE *filePointer, Image *imagesArray,
		int currentImagesCount, Image *image);

// ======================================= Service methods =======================================
void printColorsMatrix(int *colorsMatrix, int dimension);
void printImage(Image image);
void printAllData(float matchingEpsilon, int picturesCount,
		Image *picturesArray, int objectsCount, Image *objectsArray);

int isExistingImage(Image *imagesArray, int imagesCount, int imageID);
int isValidPosition(int pictureDimension, int objectDimension, Position position);
int isNotEmptyPosition(Position position);

// ======================================= Output methods =======================================
void writeOutputFile(SearchRecord *searchRecords, int recordsCount);
