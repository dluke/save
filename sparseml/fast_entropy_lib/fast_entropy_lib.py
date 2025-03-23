
###############################################################################
###############################################################################
#
#   fastEntropyLib: Python library with fast algorithms for the computation of:
#       approximate entropy
#       sample_entropy
#       bubble entropy
###############################################################################
###############################################################################

import entropy_definitions

###############################################################################


def approximate_entropy(x, m=2, r=0.2, algorithm='bucket', rsplit=5):
    """
        x: the input time series
        m: the embedding dimension
            typical and default value: 2
        r: the threshold similarity distance
           typical and default value: 0.2
        algorithm: selects which algorithm will be used
                   choose from: 'straighforward', 'bucket', 'lightweight'
                   default value = 'bucket'
        rsplit: used only by the bucket_assisted algorithm
                integer value that affects the computation time
                suggested and default value: 5
        return value: the approximate entropy
    """
    return entropy_definitions.approximate_entropy.approximate_entropy(x, m, r, rsplit, algorithm)

###############################################################################


def sample_entropy(x, m=2, r=0.2, algorithm='bucket', rsplit=5):
    """
        x: the input time series
        m: the embedding dimension
            typical and default value: 2
        r: the threshold similarity distance
           typical and default value: 0.2
        algorithm: selects which algorithm will be used
                   choose from: 'straighforward', 'bucket', 'lightweight'
                   default value = 'bucket'
        rsplit: used only by the bucket_assisted algorithm
                integer value that affects the computation time
                suggested and default value: 5
        return value: the sample entropy
    """
    return entropy_definitions.sample_entropy.sample_entropy(x, m, r, rsplit, algorithm)

###############################################################################


def bubble_entropy(x, m):
    """
        x: the input time series
        m: the embedding dimension
        return value: the bubble entropy
    """
    return entropy_definitions.bubble_entropy.bubble_entropy(x, m)

###############################################################################
###############################################################################































