# ImageSearcher

A project of identifying objects in pictures in a parallel way, using MPI, OpenMP and CUDA.

![image](https://user-images.githubusercontent.com/62587988/213797162-0b301f39-6ed4-44b6-8fba-270c0038c45e.png)

## Authors

- [Ran Yunger](https://github.com/RanYunger)

## Modus Operandi

Given |P| pictures of size MxM and |O| objects of size NxN:
- MPI: runs 2 <= x <= |P| processes, a master and at least one slave.
  -  The master loads the pictures and objects from file and manages workshares to its slaves.
  -  Each slave searches all objects in its designated picture.
  
  ![image](https://user-images.githubusercontent.com/62587988/213802847-1f2e69cb-7871-4472-ab79-0ed7ce83e3df.png)
  
- OpenMP: Each MPI slave process creates an OpenMP thread for each object and searches all the objects in the picture simultaniously.

  ![image](https://user-images.githubusercontent.com/62587988/213803898-98c0e5df-cd8e-4b8e-9ed3-2d4d34879730.png)
  
- CUDA: Each OpenMP thread turns to the GPU and calculates the matching value on each position the object can be found in.

