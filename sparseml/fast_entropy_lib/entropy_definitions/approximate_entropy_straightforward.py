
import itertools
import numpy

##########################################################################


def embed(x,m):
    '''
    embeds a signal into dimension m
    '''
    X=[]
    for i in range(len(x)-m+1):
        t=[]
        for j in range(i,i+m):
            t.append(x[j])
        X.append(numpy.array(t))
    return X
    

########################################################################## 


def norm(p1,p2,r):
    '''
    checks if p1 is similar to p2
    '''
    for i,j in zip(p1,p2):
        if numpy.abs(i-j)>r:
            return 0
    return 1


##########################################################################
    

def fm(timeseries,m,r):
    ''' 
    entopy in dimension m
    '''

    r=r*numpy.std(timeseries)
   
    X=embed(timeseries,m)

    N=len(X) 
    cm = [1]*N # 1 is for self-matching

    for i,p in enumerate (itertools.permutations(X,2),0):
        cm[i//(N-1)] += norm(p[0],p[1],r)
    
    fm = 0
    for x in cm:
        fm += numpy.log(x/N)

    return fm/N


########################################################################## 


def approximate_entropy_straightforward(timeseries,m=2,r=0.2):

    '''
    a slow straightforward implementation of approximate entropy
    '''
    
    fm0 = fm(timeseries=timeseries,m=m,r=r)
    fm1 = fm(timeseries=timeseries,m=m+1,r=r)
    
    return fm0-fm1
    

##########################################################################
##########################################################################

