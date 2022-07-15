// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "mpi.h"
#include "Utilities.h"

int main(int argc, char *argv[])
{
	// All processes' MPI variables
	int processCount, processID;
	SearchRecord slaveSearchRecord;

	// All processes' general variables
	float matchingEpsilon;
	int objectsCount, sizeInBytes;
	char *packedObjectsArray;
	Image picture, *objectsArray;

	// MPI initiation - gets amount of MPI processes and gives each a unique rank
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &processID);
	MPI_Comm_size(MPI_COMM_WORLD, &processCount);
	if (processCount < 2)
	{
		printf("Run the program with at least 2 processes\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(0);
	}

	if (processID == 0)
	{
		// MPI variables
		int picturesCount, sentPictureOffset;
		MPI_Status status;

		// General variables
		Image *picturesArray;
		SearchRecord *searchRecordsArray;
		double startTime, endTime;

		// Reads all data from the input file
		readInputFile(&matchingEpsilon, &picturesCount, &picturesArray, &objectsCount, &objectsArray);
		
		// Prints all data (for debugging purposes)
		printAllData(matchingEpsilon, picturesCount, picturesArray, objectsCount, objectsArray);
					
		startTime = MPI_Wtime();

		// Prevents idle processes
		if (processCount > picturesCount)
		{
			printf("Run the program with up to %d (# pictures) processes\n",
					picturesCount);
			MPI_Abort(MPI_COMM_WORLD, 0);
			exit(0);
		}

		// Allocates memory for the search records array
		searchRecordsArray = (SearchRecord*) malloc(picturesCount * sizeof(SearchRecord));
		if (searchRecordsArray == NULL)
		{
			printf("\nCannot allocate memory for search records\n");
			MPI_Abort(MPI_COMM_WORLD, 0);
			exit(0);
		}

		// Broadcasts the matching epsilon to all processes
		MPI_Bcast(&matchingEpsilon, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Broadcasts the objects array's size in bytes to all processes (for memory allocation)
		sizeInBytes = countIntegersInImagesCollection(objectsArray,
				objectsCount) * sizeof(int);
		MPI_Bcast(&sizeInBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Packs the objects array and broadcasts it to all processes
		packedObjectsArray = packImagesCollection(objectsArray, objectsCount);
		MPI_Bcast(packedObjectsArray, sizeInBytes, MPI_CHAR, 0, MPI_COMM_WORLD);

		// Sends each process its initial picture to handle
		for (sentPictureOffset = 0; sentPictureOffset < (processCount - 1);
				sentPictureOffset++)
		{
			sendPictureToProcess(picturesArray[sentPictureOffset],
					sentPictureOffset + 1);
			printf("Master sent picture %d to slave %d\n", picturesArray[sentPictureOffset].ID, sentPictureOffset + 1);
		}

		// While there are pictures to handle
		while (sentPictureOffset < picturesCount)
		{
			// Receives a search record from any available process
			MPI_Recv(&slaveSearchRecord, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			searchRecordsArray[slaveSearchRecord.pictureID - 1] = slaveSearchRecord;
					
			printf("Master recieved search record for picture %d from slave %d\n", slaveSearchRecord.pictureID, status.MPI_SOURCE);

			// Sends the next picture to the available process
			sendPictureToProcess(picturesArray[sentPictureOffset++], status.MPI_SOURCE);
					
			printf("Master sent picture %d to slave %d\n", picturesArray[sentPictureOffset - 1].ID, status.MPI_SOURCE);
		}

		// Waits for all slaves to finish their works
		for (int i = 1; i < processCount; i++)
		{
			// Receives final search record from a slave
			MPI_Recv(&slaveSearchRecord, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

			searchRecordsArray[slaveSearchRecord.pictureID - 1] = slaveSearchRecord;

			// Sends termination tag to the slave to kill it
			MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, TERMINATE, MPI_COMM_WORLD);
			
			printf("Master sent termination tag to slave %d\n", status.MPI_SOURCE);
		}

		// Writes search results to the output file
		writeOutputFile(searchRecordsArray, picturesCount);

		endTime = MPI_Wtime();

		printf("Master terminated successfully [Time (seconds): %.6f]\n", endTime - startTime);

	}
	else
	{
		// General variables
		SearchRecord searchRecord;

		// Receives the matching epsilon from master's broadcast
		MPI_Bcast(&matchingEpsilon, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

		// Receives packed objects array's size in bytes from master's broadcast
		// and allocates memory for the packed objects array
		MPI_Bcast(&sizeInBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
		packedObjectsArray = (char*) malloc(sizeInBytes * sizeof(char));
		if (packedObjectsArray == NULL)
		{
			printf("\nCannot allocate memory for slave %d packed objects array\n", processID);
			MPI_Abort(MPI_COMM_WORLD, 0);
			exit(0);
		}

		// Receives and unpacks packed objects array from master's broadcast
		MPI_Bcast(packedObjectsArray, sizeInBytes, MPI_CHAR, 0, MPI_COMM_WORLD);
		objectsArray = unpackImagesCollection(packedObjectsArray, &objectsCount);

		// While the master sends pictures to handle
		picture = recievePictureFromProcess(0);
		while (picture.ID != NOTFOUND)
		{
			printf("Slave %d received picture %d from master\n", processID, picture.ID);

			// Searches the picture and sends the search record to the master
			searchRecord = searchObjectsInPicture(picture, objectsArray, objectsCount, matchingEpsilon);
					
			MPI_Send(&searchRecord, 4, MPI_INT, 0, WORK, MPI_COMM_WORLD);
			
			printf("Slave %d sent search record for picture %d to master\n", processID, picture.ID);

			// Receives the next picture to handle
			picture = recievePictureFromProcess(0);
		}

		printf("Slave %d terminated successfully\n", processID);
	}

	// MPI finalization - terminates all running MPI processes
	MPI_Finalize();
	
	return 0;
}
