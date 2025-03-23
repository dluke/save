
import itertools
import numpy

#############################################################################


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
    

############################################################################# 


def norm(p1,p2,r):
    '''
    checks if p1 is similar to p2
    '''
    for i,j in zip(p1,p2):
        if numpy.abs(i-j)>r:
            return 0
    return 1


############################################################################# 


def double_norm(p1,p2,r):
    '''
    checks if p1 is similar to p2
    and also if p1[:-1] is similar to p2[:-1]
    '''
    for i in range(len(p1)-1):
        if numpy.abs(p1[i]-p2[i])>r:
            return 0,0
    if numpy.abs(p1[-1]-p2[-1])>r: 
        return 1,0
    return 1,1


#############################################################################    


def sample_entropy_straightforward(timeseries,m=2,r=0.2):

    '''a slow straightforward implementation of sample entropy
    '''
    r=r*numpy.std(timeseries)
   
    X=embed(timeseries,m+1)
    
    countSimA=0
    countSimB=0    
    for p in itertools.combinations(X,2):
        norm1,norm2 = double_norm(p[0],p[1],r)
        countSimB+=norm1
        countSimA+=norm2
        
    return -numpy.log(countSimA/countSimB)
    
    
#############################################################################
