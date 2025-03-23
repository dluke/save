

from numpy import histogram, log


############################################

def bubble_count(x):
    """
    counts the number of swaps when sorting
    :param x: the input vector
    :return: the total number of swaps
    """
    y = 0
    for i in range(len(x) - 1, 0, -1):
        for j in range(i):
            if x[j] > x[j + 1]:
                x[j], x[j + 1] = x[j + 1], x[j]
                y += 1
    return y


############################################

def complexity_count_fast(x, m):
    """
    :param x: the input series
    :param m: the dimension of the space
    :return: the series of complexities for total number of swaps
    """

    if len(x) < m:
        return []

    y = [bubble_count(x[:m])]
    v = sorted(x[:m])

    for i in range(m,len(x)):
        steps = y[i-m]
        steps -= v.index(x[i-m])
        v.pop(v.index(x[i-m]))
        v.append(x[i])
        j = m-1
        while j > 0 and v[j] < v[j-1]:
            v[j], v[j-1] = v[j-1], v[j]
            steps += 1
            j -= 1
        y.append(steps)

    return y


############################################

def renyi_int(data):
    """
    returns renyi entropy (order 2) of an integer series and bin_size=1
    (specified for the needs of bubble entropy)
    :param data: the input series
    :return: metric
    """
    counter = [0] * (max(data) + 1)
    for x in data:
        counter[x] += 1
    r = 0
    for c in counter:
        p = c / len(data)
        r += p * p
    return -log(r)


########################################


def bubble_entropy(x, m=10):
    """
    computes bubble entropy following the definition
    :param x: the input signal
    :param m: the dimension of the embedding space
    :return: metric
    """
    complexity = complexity_count_fast(x, m)
    B = renyi_int(complexity) / log(1+m*(m-1)/2)

    complexity = complexity_count_fast(x, m+1)
    A = renyi_int(complexity) / log(1+(m+1)*m/2)

    return (A-B)


def bubble_entropy_2(x, m=10):
    """
    computes bubble entropy following the definition
    :param x: the input signal
    :param m: the dimension of the embedding space
    :return: metric
    """
    complexity = complexity_count_fast(x, m)
    B = renyi_int(complexity) / log(1+m*(m-1)/2)

    complexity = complexity_count_fast(x, m+2)
    A = renyi_int(complexity) / log(1+(m+2)*(m+1)/2)

    return (A-B)


########################################
########################################


