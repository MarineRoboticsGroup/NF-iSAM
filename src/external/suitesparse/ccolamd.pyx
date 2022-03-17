import numpy as np
import scipy.sparse as sp
from typing import List
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc, free


cdef extern from "ccolamd.h":
    int ccolamd_l_recommended(int nnz, int n_row, int n_col)
    int ccolamd(int n_row, int n_col, int A_len, int *A, int *p,
    	    	double knobs [20], int stats [20], int *cmember)


def pyccolamd(S: sp.csc.csc_matrix, cmember: List[int]) -> List[int]:
    """
    Generate the Cholesky ordering for matrix S using ccolamd algorithm
    :param S: the transpose of incidence matrix to be analyzed
    :param cmember: the constraints of all variables
        each element is an integer, variables with larger values are elimated
        later
    :return: the list of indices specifying the elimination ordering
    """
    if len(S.shape) != 2:
        raise ValueError("Input argument S must be a column-based sparse "
                         "matrix")

    cdef int n_row, n_col, num_nz_elems, A_len
    n_row, n_col = S.shape
    num_nz_elems = S.count_nonzero()
    A_len = ccolamd_l_recommended(num_nz_elems, n_row, n_col)

    cdef int * A = <int *> malloc(sizeof(int) * A_len)
    cdef int * p = <int *> malloc(sizeof(int) * (n_col + 1))

    cdef int i, j
    for i in range(num_nz_elems):
        A[i] = S.indices[i]

    for j in range(n_col + 1):
        p[j] = S.indptr[j]

    cdef double * knobs = NULL
    cdef int stats[20]
    cdef int* p_cmember = <int *> malloc(sizeof(int) * n_col)

    if len(cmember) != n_col:
        raise ValueError("Input argument cmember must have the same number as "
                         "the number of variables")
    for j in range(n_col):
        p_cmember[j] = cmember[j]

    ccolamd(n_row, n_col, A_len, A, p, knobs, stats, p_cmember)

    permutation = [p[j] for j in range(n_col)]
    permutation = None

    free(A)
    free(p)
    free(p_cmember)

    return permutation
