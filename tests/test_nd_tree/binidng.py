import numpy as np
from ctypes import *
import ctypes

if __name__ == "__main__":
    testlib = ctypes.CDLL('./lib.so')

    n = 500
    dtype = np.float64
    input_array = np.array(np.linspace(0, 4 * np.pi, n), dtype=dtype)
    input_ptr = input_array.ctypes.data_as(POINTER(c_double))

    testlib.loop.argtypes = (POINTER(c_double), c_int)
    testlib.loop.restype = POINTER(c_double * n)
    testlib.freeArray.argtypes = POINTER(c_double * n),

    result_ptr = testlib.loop(input_ptr, n)
    result_array = np.frombuffer(result_ptr.contents)

    # ...do some processing
    for value in result_array:
        print(value)

    # free buffer
    testlib.freeArray(result_ptr)