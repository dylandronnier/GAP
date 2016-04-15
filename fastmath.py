import numba

@numba.jit(nopython=True)
def euclidian_norm(v):
        return inner_product(v, v) ** 0.5

@numba.jit(nopython=True)
def norm3d(v):
        assert len(v) == 3
return euclidian_norm(v)
        
@numba.jit(nopython=True)
def inner_product(v1, v2):
        n = len(v1)
        assert n == len(v2)
	
        r = 0.0
        for i in xrange(0, n):
        r += v1[i] * v2[i]
        return r
