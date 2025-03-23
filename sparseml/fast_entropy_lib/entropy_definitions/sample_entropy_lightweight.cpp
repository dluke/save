
// g++ -Wall -O2 -shared -fPIC sample_entropy_lightweight.cpp -o sample_entropy_lightweight_lib.so 

#include <stdlib.h>
#include <math.h>


typedef struct {
	double Dato;
	unsigned int p_ini;
} d_o;

int sort_function( const void *a, const void *b );


extern "C" double light(double *x, int N, int m, double r)
{
	int Nm = N-m;
    int i,j,k,ii,jj;
    int A=0,B=0;
	int r_sup, i_inf, i_sup, i_mez;
	double *X;
	double D_piu_r;
	d_o  *_D_ordinati;
	int *originalPositions;


    double sum=0.0;
    for(i=0; i<N; i++)
        sum += x[i];
    double mean = sum/N;        
    double standardDeviation=0.0;
    for (i=0; i<N; i++)
        standardDeviation += (x[i]-mean)*(x[i]-mean);
    standardDeviation = sqrt(standardDeviation/N);
    r=r*standardDeviation;
    
	
	X = (double *)malloc(Nm*sizeof(double));
	for (i=0;i<Nm;i++) 
	{
		X[i]=x[i];
		for (j=1;j<m;j++)
			X[i]+=x[i+j];
	}
	
	_D_ordinati = (d_o *)malloc(Nm*sizeof(d_o));
	for(i=0; i<Nm; i++)
	{
		_D_ordinati[i].Dato=X[i];
		_D_ordinati[i].p_ini=i;
	}
	
	qsort((void *)_D_ordinati, (size_t)Nm, sizeof(d_o), sort_function);

	originalPositions = (int *)malloc(Nm*sizeof(int));
	
	for(i=0; i<Nm; i++)
	{
		X[i]=_D_ordinati[i].Dato;
		originalPositions[i]=_D_ordinati[i].p_ini;
	}

	free(_D_ordinati);
	
	for(i=0; i<Nm; i++)
	{
		D_piu_r=X[i]+m*r;
		if(D_piu_r >= X[Nm-1])
			r_sup=Nm-1;
		else 
		{
			i_inf=i;
			i_sup=Nm-1;
			while(i_sup-i_inf>1) 
			{
				i_mez=(i_inf+i_sup)>>1;
				if( X[i_mez] > D_piu_r )
					i_sup=i_mez;
				else
					i_inf=i_mez;
			}
			r_sup=i_inf;
		}
		ii=originalPositions[i];
		for(j=i+1; j<=r_sup; j++) 
		{
			jj=originalPositions[j];
			
			for (k=0;k<m;k++)
				if (fabs(x[ii+k]-x[jj+k])>r)
					break;
			if (k==m)
			{
				B++;
				if (fabs(x[ii+m]-x[jj+m])<=r)
					A++;
			}
		}    
	}

    if (A*B==0)
         return log((N-m)*(N-m-1));
    else
        return -log(1.0*A/B);    
}


int sort_function( const void *a, const void *b)
{
	return ( ((d_o *)a)->Dato > ((d_o *)b)->Dato ) ? 1 : -1;
}


