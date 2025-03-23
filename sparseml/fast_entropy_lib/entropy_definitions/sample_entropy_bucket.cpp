
// g++ -Wall -O2 -shared -fPIC sample_entropy_bucket.cpp -o sample_entropy_bucket_lib.so

#include <stdlib.h>
#include <math.h>

#define EMPTY -1
#define ERROR -1

void swap(int *a, int *b);
void quicksort(int arr[], double data[], int beg, int end);
int binarySearchHigh(int items[], double data[], int count, double key);
int binarySearchLow(int items[], double data[], int count, double key);


extern "C" double bucket(double *x, int N, int m, double r, int rfactor)
{
    double bucket_size;
    
    int Nm = N-m;
    int i,j,k;


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
    

   	// integration
    double *X;
    if((X = (double *)malloc(Nm*sizeof(double))) == NULL)
        return ERROR;
    for (i=0;i<Nm;i++)
    {
        X[i]=x[i];       
        for (j=1;j<m;j++)
            X[i]+=x[i+j];
    }
	
 
   
	//normalization: X from 0 to Xmax
    double Xmin,Xmax; Xmin = Xmax = X[0];
    for (i=1;i<Nm;i++)
    {
        if (Xmin>X[i]) Xmin=X[i];  
        if (Xmax<X[i]) Xmax=X[i];
    }   
    for (i=0;i<Nm;i++)
        X[i]-=Xmin;      
    Xmax-=Xmin;


    // init buckets
	bucket_size = r/rfactor; // size of bucket
	int Nb = (int)ceil(Xmax/bucket_size); // number of buckets
    int *bucket; // the first element in the bucket
    int *last_in_bucket; // the last element in the bucket
    int *list; // connection from first to last element, actually a list
    
    int p;
    if((bucket = (int *)malloc(Nb*sizeof(int))) == NULL)  
    {
        free(X);
        return ERROR;
    }
    if((last_in_bucket = (int *)malloc(Nb*sizeof(int))) == NULL)
    {
        free(X);
        free(bucket);
        return ERROR;  
    }
    if((list = (int *)malloc(Nm*sizeof(int))) == NULL) 
    {
        free(X);
        free(bucket);
        free(last_in_bucket);
        return ERROR;  
    }
    for (i=0;i<Nb;i++)
    {
        bucket[i]=EMPTY;
        last_in_bucket[i]=EMPTY;
    }
    for (i=0;i<Nm;i++)
        list[i]=EMPTY;
        

    // fill buckets
    for (i=0;i<Nm;i++)
    {
        p=(int)floor(X[i]/bucket_size);       
        if (last_in_bucket[p]==EMPTY)
        {
            bucket[p]=i;
            last_in_bucket[p]=i;
        }
        else
        {
            list[last_in_bucket[p]]=i;
            last_in_bucket[p]=i;
        }
    }


	// compute A,B
    int A=0; int B=0;
    int current,other; // index to bucket
    int p1,p2; // pointer to bucket
    int counter1,counter2; // pointer to elements inside the buckets, used for list1,list2
    int *list1, *list2; // lists to hold the elements to be checked for similarity
    int low,high; // limits for searching in buckets
    

	// malloc lists
    if((list1 = (int *)malloc((Nm)*sizeof(int))) == NULL)
    {
        free(X);
        free(bucket);
        free(last_in_bucket);
        free(list);
        return ERROR;  
    }
    if((list2 = (int *)malloc((Nm)*sizeof(int))) == NULL)
    {
        free(X);
        free(bucket);
        free(last_in_bucket);
        free(list);
        free(list1);
        return ERROR;  
    }
    
    for (current=0;current<Nb;current++)
    {
        p1=bucket[current];
        counter1=0;
        while (p1!=EMPTY)
        {
            list1[counter1]=p1;
            counter1++;
            p1=list[p1];
        }
        
        if (counter1!=0)
        {
            for (other=current-m*rfactor;other<current;other++)
            {
                if (other>=0)
                {
                    p2=bucket[other];
                    counter2=0;
                    while (p2!=EMPTY)
                    {
                        list2[counter2]=p2;
                        counter2++;
                        p2=list[p2];
                    }
                    if (counter2!=0)
                    {
						quicksort(list2,x,0,counter2);
                        for (i=0;i<counter1;i++)
                        {
							low=binarySearchLow(list2,x,counter2,x[list1[i]]-r);
							high=binarySearchHigh(list2,x,counter2,x[list1[i]]+r); 
							for (j=low;j<=high;j++)
                            {
								double *pi=&(x[list1[i]]); // use pointers to access the data
								double *pj=&(x[list2[j]]); //// instead of indirect access 
                                for (k=0;k<m;k++)
                                {
									if (fabs(*pi-*pj)>r)
										break;
									pi++;
									pj++;
								}
                                if (k==m)
                                {
                                    B++;
                                    if (fabs(*pi-*pj)<=r)
                                        A++;
                                }
                            }
						}
					}
                }
            }
            for (i=0;i<counter1;i++)
                for (j=i+1;j<counter1;j++)
                {
					double *pi=&(x[list1[i]]);
					double *pj=&(x[list1[j]]);
                    for (k=0;k<m;k++)
                    {
						if (fabs(*pi-*pj)>r)
							break;
						pi++;
						pj++;
					}
                    if (k==m)
                    {
						B++;
                        if (fabs(*pi-*pj)<=r)
							A++;
                    }
                }
        }
    }


	// free lists
    free(X);
    free(bucket);
    free(last_in_bucket);
    free(list);
    free(list1);
	free(list2);
    
		
	// return sampen
	if (A*B==0)
         return log((N-m)*(N-m-1));
    else
        return -log(1.0*A/B);    

}   
    
    
    
void swap(int *a, int *b)
{
	int t=*a; *a=*b; *b=t;
}

void quicksort(int arr[], double data[], int beg, int end)
{
	if (end > beg + 1)
		//	if (end > beg)
	{
		double piv = data[arr[beg]];
		int l = beg + 1, r = end;
		while (l < r)
		{
			if (data[arr[l]] <= piv)
				l++;
			else
				swap(&arr[l], &arr[--r]);
		}
		swap(&arr[--l], &arr[beg]);
		quicksort(arr, data, beg, l);
		quicksort(arr, data, r, end);
	}
}

int binarySearchHigh(int items[], double data[], int count, double key)
{
	int low, high, mid;

	low = 0; high = count-1;
	while(low <= high)
	{
		mid = (low+high)/2;
		if(key < data[items[mid]]) 
			high = mid-1;
		else if(key > data[items[mid]]) 
			low = mid+1;
		else 
		{
			while(mid+1<count && data[items[mid+1]]==key) mid++;
			return mid; /* found */
		}
	}
	return high;
}

int binarySearchLow(int items[], double data[], int count, double key)
{
	int low, high, mid;

	low = 0; high = count-1;
	while(low <= high)
	{
		mid = (low+high)/2;
		if(key < data[items[mid]]) 
			high = mid-1;
		else if(key > data[items[mid]]) 
			low = mid+1;
		else 
		{
			while(mid-1>=0 && data[items[mid-1]]==key) mid--;
			return mid; /* found */
		}
	}
	return low;
}

