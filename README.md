# ImageSearcher

A project of identifying objects in pictures in a parallel way, using MPI, OpenMP and CUDA.

![image](https://user-images.githubusercontent.com/62587988/213797162-0b301f39-6ed4-44b6-8fba-270c0038c45e.png)

## Authors

- [Ran Yunger](https://github.com/RanYunger)

## Modus Operandi

Given |P| pictures of size MxM and |O| objects of size NxN:
- MPI: runs 2 <= x <= |P| processes, a master and at least one slave.
  The master loads the pictures and objects from file and manages workshares to its slaves, when each slave gets its own picture and copies of all objects to search in it.
  
  ![image](https://user-images.githubusercontent.com/62587988/213802847-1f2e69cb-7871-4472-ab79-0ed7ce83e3df.png)
  
- OpenMP: Each slave process creates a thread for each object and activates search 
