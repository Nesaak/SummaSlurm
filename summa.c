/*
* 	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include <string.h>
#include "mpi.h"

#define min(a, b) ((a < b) ? a : b)

// Each matrix of entire A, B, and C is SZ by SZ. 
// Uncomment any preffered size
//#define SZ 400
#define SZ 2520
//#define SZ 5040
//#define SZ 10080
//#define SZ 40320


/*Copied from CUDA CUBLAS Benchmarks*/
void matrixMulCPU(double **C, double **A, double **B, int hA, int wA, int wB) {
    for (int i = 0; i < hA; ++i)
        for (int j = 0; j < wB; ++j) {
            double sum = 0;
            for (int k = 0; k < wA; ++k) {
                double a = A[i][k];
                double b = B[k][j];
                sum += a * b;
            }
            C[i][j] = sum;
        }
}

/*Copied from CUDA CUBLAS Benchmarks*/
void printDiff(double **data1, double **data2, int width, int height, int iListLength, float fListTol) {
    printf("Listing first %d Differences > %.6lf...\n", iListLength, fListTol);
    int i,j;
    int error_count=0;
    for (j = 0; j < height; j++) {
		// Removed so output is not spammed with thousands of rows
        // if (error_count < iListLength) printf("\n  Row %d:\n", j);
        
        for (i = 0; i < width; i++) {
            double fDiff = fabs(data1[j][i] - data2[j][i]);
            if (fDiff > fListTol) {	
                if (error_count < iListLength) {
                    printf("    Loc(%d,%d)\tCPU=%.5lf\tMyRes=%.5lf\tDiff=%.6lf\n", i, j,data1[j][i], data2[j][i], fDiff);
                }
                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

/* Debug, print the matrix */
void printMatrix(double **A, int n) {
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			printf("%.4lf  ",A[i][j]);
		}
		printf("\n");
	}
}

/** 
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
        array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
        for (i=1; i<n_rows; i++){
                array[i] = array[0] + i * n_cols;
        }
        return array;
}
/** 
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}


/* Implementation Code Start */

/**
*	Perform the SUMMA matrix multiplication. 
*/
// implementation of mult C = C + A x B
void SUMMA(double **C, double **A, double **B, int n){
	int i,j,k;
	for(k=0;k<n;k++) for(i=0;i<n;i++) for(j=0;j<n;j++) C[i][j]+=A[i][k]*B[k][j];
}


/**
*	Implement matrix multiplication
*/
void matmul(int myrank, int proc_grid_sz, int block_sz, double **my_A, double **my_B, double **my_C) {
	// Define variables
	MPI_Comm grid_comm;
	int dimsizes[2];
	int wraparound[2];
	int coordinates[2];
	int free_coords[2];
	int reorder = 1;
	int my_grid_rank, grid_rank;
	int row_test, col_test;

	MPI_Comm row_comm;
	MPI_Comm col_comm;

	dimsizes[0] = dimsizes[1] = proc_grid_sz;
	wraparound[0] = wraparound[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_grid_rank);
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);

	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm, free_coords, &row_comm);
 	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm,free_coords, &col_comm);

	int x_sz;
	double **buffA, **buffB;
	buffA = alloc_2d_double(block_sz,block_sz);
	buffB = alloc_2d_double(block_sz,block_sz);
	for(int k = 0; k < proc_grid_sz; k++){
	if (coordinates[1] == k)
		memcpy(*buffA,*my_A,block_sz*block_sz*sizeof(double));
	MPI_Bcast(*buffA, block_sz*block_sz, MPI_DOUBLE, k, row_comm); //broadcast buffA from (j,k) to row j  where j = 0..proc_grid_sz
	if(coordinates[0] == k)
		memcpy(*buffB,*my_B,block_sz*block_sz*sizeof(double));
	MPI_Bcast(*buffB, block_sz*block_sz, MPI_DOUBLE, k, col_comm); // broadcast buffB from (k,j) to column j where j = 0..proc_grid_sz
	
		
		  
	if (coordinates[0]==k && coordinates[1]==k)
		SUMMA(my_C,my_A,my_B,block_sz);  // grid (k,k) buffA==my_A buff_B==my_B
	else if(coordinates[0]==k)
		SUMMA(my_C,buffA,my_B,block_sz); //grid (k,j) buffB == my_B 
	else if (coordinates[1])
		SUMMA(my_C,my_A,buffB,block_sz); // grid (j,k) buffA == my_A
	else
		SUMMA(my_C,buffA,buffB,block_sz);
	  
	}

	free(*buffA);
	free(*buffB);
	free(buffA);
	free(buffB);
}

/*

MAIN

*/
int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides
	MPI_Status status;

	srand(time(NULL));							// Seed random numbers

	// insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank

	MPI_Init(&argc, &argv);               		// Initialize MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 		// Get rank of current process
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);   	// Get total number of processes

	// assign values to 1) proc_grid_sz and 2) block_sz
	
	proc_grid_sz = (int)sqrt(num_proc); 	// 1) proc_grid_sz = q = sqrt(np),   since np = q^2
	if (proc_grid_sz != sqrt(num_proc)){
		printf("*****\nERROR: num_proc need to be q^2\n*****\n");
		exit(1);
	}
	block_sz = SZ / proc_grid_sz; //2) get block_sz   block_sz = n/q

	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);

	initialize(A, B, C, block_sz);

	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();

	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);

	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();

	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;

	// Send mat to processer 0
	if(rank!=0)
	{
		MPI_Send(&A[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+100,MPI_COMM_WORLD);
		MPI_Send(&B[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+200,MPI_COMM_WORLD);
		MPI_Send(&C[0][0],block_sz*block_sz,MPI_DOUBLE,0,rank+300,MPI_COMM_WORLD);
 	}

	// For root only
	if(rank == 0)
	{
		double **matrixA;
		double **matrixB;
		double **matrixC;

		matrixA = alloc_2d_double(SZ, SZ);
		matrixB = alloc_2d_double(SZ, SZ);
		matrixC = alloc_2d_double(SZ, SZ);
		for(int i = 0; i < num_proc; i++){
			double *tempA = (double *)malloc(block_sz*block_sz*sizeof(double));
			double *tempB = (double *)malloc(block_sz*block_sz*sizeof(double));
			double *tempC = (double *)malloc(block_sz*block_sz*sizeof(double));
			if(i!=0){
				MPI_Recv(tempA,block_sz*block_sz,MPI_DOUBLE,i,i+100,MPI_COMM_WORLD,&status);
				MPI_Recv(tempB,block_sz*block_sz,MPI_DOUBLE,i,i+200,MPI_COMM_WORLD,&status);
				MPI_Recv(tempC,block_sz*block_sz,MPI_DOUBLE,i,i+300,MPI_COMM_WORLD,&status);
			}else{
			memcpy(tempA,*A,block_sz*block_sz*sizeof(double));
			memcpy(tempB,*B,block_sz*block_sz*sizeof(double));
			memcpy(tempC,*C,block_sz*block_sz*sizeof(double));
			}
			int p = 0;
			for(int j=0; j < block_sz; j++){
				for(int k = 0 ; k < block_sz; k++){
					matrixA[j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz]=tempA[p];
					matrixB[j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz]=tempB[p];
					matrixC[j+(int)(i/proc_grid_sz)*block_sz][k+i%proc_grid_sz*block_sz]=tempC[p];
					p++;
				}
			}
			free(tempA);
			free(tempB);
			free(tempC);
		}

		// Insert statements for testing
/*
		// Create and zero out CPU solution matrix
		double **solution;
    	solution = alloc_2d_double(SZ,SZ);
    	for(int j = 0; j < SZ; j++)for(int k = 0; k < SZ; k++)solution[j][k]=0.0;

		// Compute CPU timing for comparision
		printf("Computing result using CPU...\n");
		double start_time_One_CPU, end_time_One_CPU, total_time_One_CPU;
		start_time_One_CPU = MPI_Wtime();
        matrixMulCPU(solution, matrixA, matrixB, SZ, SZ, SZ);
		end_time_One_CPU = MPI_Wtime();
		total_time_One_CPU = end_time_One_CPU - start_time_One_CPU;
		printf("total_time_One_CPU: %lf\n", total_time_One_CPU);

		printDiff(solution, matrixC, SZ, SZ, 100, 1.0e-2f);
		// printMat(matrixC, SZ);
		*/

		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n", SZ, num_proc, total_time);


	}

	// Destroy MPI processes
	MPI_Finalize();

	return 0;
}
