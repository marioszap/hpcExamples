// Exercise 1, question 3: initial code

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

void do_work(int i, int rank) {
	printf("p%d processing %d\n", rank, i);
	sleep(5);
}

int main(int argc, char** argv) {
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
		printf("Running with %d MPI processes\n", size);

	int M = 10;	// two tasks per process
	int input;

	if(rank == 0) {
		int w_size = size-1;
		int N = M*w_size;

		srand48(time(0));
		printf("N: %d\n\n", N);
		for(int i=0; i<N; i++) {
			input = lrand48() % 1000;	// some random value
			MPI_Send(&input, 1, MPI_INT, i%w_size+1, 100, MPI_COMM_WORLD);
			printf("\tsent input: %d to process: %d\n",input, i%w_size+1);
		}
	} 
	else 
		for(int i = 0; i < M; i++) {
			MPI_Recv(&input, 1, MPI_INT, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			do_work(input, rank);
		}
	
	MPI_Finalize();
	return 0;
}
