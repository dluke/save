
// g++ -Wall -O2 -shared -fPIC approximate_entropy_lightweight.cpp -o approximate_entropy_lightweight_lib.so 

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define ERROR -1


// structure which will be used for sorting
typedef struct {
	double Dato;
	unsigned int p_ini;
} d_o;


// sorting
int sort_function( const void *a, const void *b)
{
	return ( ((d_o *)a)->Dato > ((d_o *)b)->Dato ) ? 1 : -1;
}


extern "C" double light (double *x, int N, int m, double r)
{
	int Nm, i, j, k, *cm, *cm1, ii, jj;
	int r_sup, i_inf, i_sup, i_mez;
	double *X;
	double  fm, fm1;
	double D_piu_r;
	d_o  *_D_ordinati;
	int *originalPositions;

	Nm = N-m+1;
	
    //value of r
    double sum=0.0;
    for(i=0; i<N; i++)
        sum += x[i];
    double mean = sum/N;        
    double standardDeviation=0.0;
    for (i=0; i<N; i++)
        standardDeviation += (x[i]-mean)*(x[i]-mean);
    standardDeviation = sqrt(standardDeviation/N);
    r=r*standardDeviation;


	if((X = (double *)malloc(Nm*sizeof(double))) == NULL)
		return ERROR;
	if((cm = (int *)malloc(Nm*sizeof(int))) == NULL) {
		free(X);
		return ERROR;
	}
	if((cm1 = (int *)malloc((Nm-1)*sizeof(int))) == NULL) {
		free(X);
		free(cm);
		return ERROR;
	}

	// Prepare the structure which will be used for sorting
	if((_D_ordinati = (d_o *)malloc(Nm*sizeof(d_o))) == NULL) {
		free(X);
		free(cm);
		free(cm1);
		return ERROR;
	}
	for(i=0; i<Nm; i++) {
		_D_ordinati[i].Dato=x[i];
		_D_ordinati[i].p_ini=i;
	}

	// Sort the data
	qsort((void *)_D_ordinati, (size_t)Nm, sizeof(d_o), sort_function);

	// an additional structure to make struct navigation quicker
	if((originalPositions = (int *)malloc(Nm*sizeof(int))) == NULL) {
		free(X);
		free(cm);
		free(cm1);
		free(_D_ordinati);
		return ERROR;
	}
	for(i=0; i<Nm; i++) {
		X[i]=_D_ordinati[i].Dato;
		originalPositions[i]=_D_ordinati[i].p_ini;
	}
	// _D_ordinati is not necessary beyond here
	free(_D_ordinati);

	// Initialize the densities
	for(i=0; i<Nm-1; i++) {
		cm[i]=1;
		cm1[i]=1;
	}
	cm[Nm-1]=1;

	// Find the potentially close-enough points
	for(i=0; i<Nm; i++) {
		D_piu_r=X[i]+r;
		if(D_piu_r >= X[Nm-1])
			r_sup=Nm-1;
		else {
			i_inf=i;
			i_sup=Nm-1;
			while(i_sup-i_inf>1) {
				i_mez=(i_inf+i_sup)>>1;
				if( X[i_mez] > D_piu_r )
					i_sup=i_mez;
				else
					i_inf=i_mez;
			}
			r_sup=i_inf;
		}

		ii=originalPositions[i];
		for(j=i+1; j<=r_sup; j++) {
			jj=originalPositions[j];
			k=1;
			while(k<m) {
				if (fabs(x[ii+k]-x[jj+k])>r) k=m;
				k++;
			}
			if(k==m) {
				cm[ii]++;
				cm[jj]++;
				if( (ii+m<N) && (jj+m<N) && fabs(x[ii+m]-x[jj+m])<=r) {
					cm1[ii]++;
					cm1[jj]++;
				}
			}
		}    
	}
	
	for (i=0;i<N-m+1;i++)
		printf("%d\n",cm[i]);
	printf("\n");
	for (i=0;i<N-m;i++)
		printf("%d\n",cm1[i]);
	printf("\n");
	
	
	// compute ApEn
	fm=0;
	for (i=0;i<N-m+1;i++)
		fm+=log(((double)cm[i])/(N-m+1));

	fm1=0;
	for (i=0;i<N-m;i++)
		fm1+=log(((double)cm1[i])/(N-m));

	fm=fm/(N-m+1);
	fm1=fm1/(N-m);
	
	free(X);
	free(cm);
	free(cm1);
	free(originalPositions);

	return fm-fm1;

}

