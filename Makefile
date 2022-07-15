build:
	mpicxx -fopenmp -c Main.c -o Main.o
	mpicxx -fopenmp -c CFunctions.c -o CFunctions.o
	nvcc -I./inc -c CudaFunctions.cu -o CudaFunctions.o
	mpicxx -fopenmp -o FinalProject Main.o CFunctions.o CudaFunctions.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./FinalProject

run:
	mpiexec -np 2 ./FinalProject

runOn2:
	mpiexec -np 2 -hostfile hosts.txt -map-by node ./FinalProject

