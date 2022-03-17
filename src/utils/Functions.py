import json

import numpy as np
import pandas

from utils.Units import _TWO_PI
from typing import Dict, List
from slam.Variables import Variable


def sort_pair_lists(number_list, attached_list) -> "sorted_number_list, sorted_attached_list":
    sorted_number_list, sorted_attached_list = (list(t) for t in zip(*sorted(zip(number_list, attached_list))) )
    return sorted_number_list, sorted_attached_list


def none_to_zero(x) -> "x":
    return 0.0 if x is None else x


def theta_to_pipi(theta):
    return (theta + np.pi) % _TWO_PI - np.pi


def sample_dict_to_array(samples: Dict[Variable, np.ndarray],
                         ordering: List[Variable] = None):
    """
    Convert samples from a dictionary form to numpy array form
    """
    if ordering is None:
        ordering = list(samples.keys())
    elif set(ordering) != set(samples.keys()):
        raise ValueError("Variables in the ordering do not match those in "
                         "the dictionary")
    return np.hstack((samples[var] for var in ordering))

def sample_from_arr(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def array_order_to_dict(samples: np.ndarray, order: List[Variable])->Dict:
    res = {}
    cur_idx = 0
    for var in order:
        res[var] = samples[:,cur_idx:cur_idx+var.dim]
        cur_idx += var.dim
    return res

def kabsch_umeyama(A, B):
    # authored by:
    # https://zpl.fi/aligning-point-patterns-with-kabsch-umeyama-algorithm/
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB
    return R, c, t

def reject_outliers(data, iq_range=0.5):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    sr = pandas.Series(data)
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return np.where(np.logical_and(data>=qlow-1.7*iqr, data<=qhigh+1.7*iqr))[0] # get 99.7% data
    # return sr[ (sr - median).abs() <= iqr]
