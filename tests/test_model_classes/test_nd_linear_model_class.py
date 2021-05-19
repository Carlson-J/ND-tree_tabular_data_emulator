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
    EPS = 10 ** -15
    # build function values
    x0 = np.linspace(-1, 1, 10)
    x1 = np.linspace(-2, 2, 7)
    X0, X1 = np.meshgrid(x0, x1)
    # compute fucntion
    F = X0 * 2 + X1
    # get weights
    weights = fit_nd_linear_model(F, [[x0[0], x1[0]], [x0[-1], x1[-1]]])
    # compare to truth
    TRUE_WEIGHTS = np.array([-4, 0, 0, 4, -1, -2, 1, 2])
    for w_true, w in zip(TRUE_WEIGHTS, weights):
        assert (abs(w - w_true) <= EPS)


def test_nd_linear_model_class_2d():
    """
    GIVEN: weight and a location to do a 2D linear interpolation of f(x) = x0*2 + x1
    WHEN: does interpolation
    THEN: correct interpolation vale
    :return:
    """
    DIMS = 2
    EPS = 10 ** -15
    # following the ordering [f_00, f_10, f_01, f_11, x0_00, x1_00, x0_11, x1_11]
    weights = np.array([-4, 0, 0, 4, -1, -2, 1, 2])
    # input points

    x = np.array([0.5, -1.5])
    F_TRUE = -0.5

    f_interp = nd_linear_model(weights, x)

    assert (abs(f_interp - F_TRUE) <= EPS)


def test_nd_linear_model_class_8d():
    """
    GIVEN: corner points of f(x) = sum(x_i*(i+1))
    WHEN: interpolate random x's for f(x) = sum(x_i*(i+1))
    THEN: interpolated values are exact to precision
    :return:
    """
    EPS = 10 ** -10
    DIMS = 8

    # Create function that is 8d-linear
    def f(x):
        return sum([x[k] * (k + 1) for k in range(DIMS)])

    # pick extreme corners
    corner1 = [-1, 2, 4, 5, -1.5, -1, -1.1, -0.9]
    corner2 = [1, 3, 6, 5.1, 7, 1, 2, 1]
    # create weight vector
    weights = np.zeros([2 ** DIMS + 2 * DIMS])
    weights[2 ** DIMS:2 ** DIMS + DIMS] = corner1
    weights[2 ** DIMS + DIMS:] = corner2
    # evaluate function at each corner of the resulting hypercube
    for i in range(2 ** DIMS):
        x = []
        for j in range(DIMS):
            x.append(corner1[j] if (i >> j) & 1 == 0 else corner2[j])
        weights[i] = f(np.array(x))

    # evaluate f at random points
    NUM_TEST_POINTS = 100
    X = np.random.rand(NUM_TEST_POINTS, DIMS)
    f_true = np.array([f(X[i, :]) for i in range(NUM_TEST_POINTS)])

    # do interpolation at points
    f_interp = np.array([nd_linear_model(weights, X[i, :]) for i in range(NUM_TEST_POINTS)])

    # check that they are correct
    assert (np.all(abs(f_interp - f_true) <= EPS))
