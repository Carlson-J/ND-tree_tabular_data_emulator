from nd_emulator.model_classes import nd_linear_model, fit_nd_linear_model
import numpy as np


def test_nd_linear_model_class_fit_2d():
    """
    GIVEN: weight and a location to do a 2D linear interpolation of f(x) = x0*2 + x1
    WHEN: does interpolation
    THEN: correct interpolation vale
    :return:
    """
    DIMS = 2
    EPS = 10**-15
    # build function values
    x0 = np.linspace(-1, 1, 10)
    x1 = np.linspace(-2, 2, 7)
    X0, X1 = np.meshgrid(x0, x1)
    # compute fucntion
    F = X0*2 + X1
    # get weights
    weights = fit_nd_linear_model(F, [[x0[0], x1[0]], [x0[-1], x1[-1]]])
    # compare to truth
    TRUE_WEIGHTS = np.array([-4, 0, 0, 4, -1, -2, 1, 2])
    for w_true, w in zip(TRUE_WEIGHTS, weights):
        assert(abs(w-w_true) <= EPS)



def test_nd_linear_model_class_2d():
    """
    GIVEN: weight and a location to do a 2D linear interpolation of f(x) = x0*2 + x1
    WHEN: does interpolation
    THEN: correct interpolation vale
    :return:
    """
    DIMS = 2
    EPS = 10**-15
    # following the ordering [f_00, f_10, f_01, f_11, x0_00, x1_00, x0_11, x1_11]
    weights = np.array([-4, 0, 0, 4, -1, -2, 1, 2])
    # input point
    x = np.array([0.5, -1.5])
    F_TRUE = -0.5

    f_interp = nd_linear_model(weights, x)

    assert(abs(f_interp - F_TRUE) <= EPS)
