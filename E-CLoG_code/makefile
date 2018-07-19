CC=/usr/bin/g++-4.7 -std=c++0x -g -O3

Link=-fopenmp

all: parallel

parallel:
	$(CC) $(Link) graphlet_counting_omp.cpp -o omp_graphlet_count_5 

clean:
	rm omp_graphlet_count_5
