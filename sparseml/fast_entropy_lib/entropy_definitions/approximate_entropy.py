
import site
import sysconfig
import ctypes
from .approximate_entropy_straightforward import approximate_entropy_straightforward


#############################################################################


def approximate_entropy(x, m=2, r=0.2, rsplit=5, algorithm='bucket'):

    for s in x:
        if not type(s) in (int, float):
            return 'error: the time series should consists of integers or floats'
    if not type(m) == int or m < 1:
        return 'error: m should me an integer, m=1,2,3,...'
    if not type(r) in (int, float) or r<=0:
        return 'error: r should me a possitive float value'
    if not type(rsplit) == int or rsplit<=0:
        return 'error: rsplit should me a possitive integer value'

    if algorithm == 'straightforward':
        return approximate_entropy_straightforward(x, m, r)
    elif algorithm == 'bucket':
        return approximate_entropy_bucket(x,m,r,rsplit)
    elif algorithm == 'lightweight':
        return approximate_entropy_lightweight(x, m, r)
    else:
        return 'error: algorithm '+algorithm+' not supported'

#######################################################################


def approximate_entropy_bucket(x,m,r,rsplit):

    site_packages = site.getsitepackages()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    lib_path = site_packages[0] + '/approximate_entropy_bucket_lib'+suffix
    lib = ctypes.cdll.LoadLibrary(lib_path)

    N = len(x)
    array_type = ctypes.c_double * N
    lib.bucket.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ctypes.c_int,
                           ]
    lib.bucket.restype = ctypes.c_double
    return lib.bucket(array_type(*x),N,m,r,rsplit)


#############################################################################


def approximate_entropy_lightweight(x,m,r):

    site_packages = site.getsitepackages()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    lib_path = site_packages[0] + '/approximate_entropy_lightweight_lib'+suffix
    lib = ctypes.cdll.LoadLibrary(lib_path)

    N = len(x)
    array_type = ctypes.c_double * N
    lib.light.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.c_int,
                           ctypes.c_int,
                           ctypes.c_double,
                           ]
    lib.light.restype = ctypes.c_double
    return lib.light(array_type(*x),N,m,r)


#############################################################################
