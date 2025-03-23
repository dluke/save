
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>  
#include <math.h>


double renyi_int(int * complexity, int N, int m)
{
	double entropy;
	int * counter;
	int counter_size;
	int i; 
	double p;
		
	counter_size = m*(m-1)/2+1;
	entropy = 0.0;
	
	counter = (int *) malloc(counter_size*sizeof(int));
	
	for (i=0; i<counter_size; i++) counter[i]=0;

	for (i=0; i<N; i++) counter[complexity[i]]++;

	for (i=0; i<counter_size; i++) 
	{
		p = (double) counter[i] / counter_size;
		entropy += p * p;
	}
	entropy = -log(entropy);

	return entropy;
}


int * complexity_estimate_fast(double * rr, int N, int m)
{
	int * complexity;
	double * vector;
	
	int i,j;
	double t;
	int swaps;
	int element_to_remove;
	
		
	// malloc
	complexity = (int *) malloc((N-m+1)*sizeof(int));
	vector = (double *) malloc(m*sizeof(double));
	
	// init
	for (i=0;i<m;i++) vector[i] = rr[i];
	for (i=0;i<N;i++)complexity[i]=0;
	
	// compute complexity[0]
	for (i=0;i<m-1;i++)
		for (j=0; j < m-i-1; j++)
			if (vector[j]>vector[j+1])
			{
				t = vector[j];
				vector[j] = vector[j+1];
				vector[j+1] = t;
				complexity[0]++;
			}
	
	// compute the rest of the complexity array
	for (i=1; i<N-m+1; i++)
	{
		swaps = complexity[i-1];
		element_to_remove = rr[i-1];
		j=0;
		while (vector[j]!=element_to_remove) j++;
		swaps -= j;
		while (j<m-1)
		{
			vector[j] = vector[j+1];
			j++;
		}
		vector[m-1] = rr[i+m-1];

		j=m-2;
		while (vector[j]>vector[j+1])
		{
			t = vector[j];
			vector[j] = vector[j+1];
			vector[j+1] = t;
			swaps++;
			j--;
		}
		complexity[i] = swaps;
	}

	return complexity;
}


double swap_entropy(double * rr, int N, int m)
{
	int * complexity;
	double entropy;
	
	complexity = complexity_estimate_fast(rr,N,m);

	entropy = renyi_int(complexity, N-m+1, m) / log(1+m*(m-1)/2);

	return entropy;
}



double bubble_entropy(double * rr, int N, int m)
{
	if (N>m) return swap_entropy(rr,N,m+1) - swap_entropy(rr,N,m); 
	else return 0;
}



int main(int argc, char **argv)
{
	double bubble;
	double * rr;
	int N;
	int m;
		
	
	N = 100000;
	rr = (double *) malloc (N * sizeof(double));
	m =3;
		
	srand(1);
    int i; for(i=0; i<N; i++) rr[i] = rand();
	bubble = bubble_entropy(rr,N,m);
	
	printf("%g\n",bubble);

	return 0;
}



